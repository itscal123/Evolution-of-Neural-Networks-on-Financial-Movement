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
bpstkcuudk/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebpstkcuudk/kernel
{
%bpstkcuudk/kernel/Read/ReadVariableOpReadVariableOpbpstkcuudk/kernel*"
_output_shapes
:*
dtype0
v
bpstkcuudk/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebpstkcuudk/bias
o
#bpstkcuudk/bias/Read/ReadVariableOpReadVariableOpbpstkcuudk/bias*
_output_shapes
:*
dtype0
~
oaettnoaty/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameoaettnoaty/kernel
w
%oaettnoaty/kernel/Read/ReadVariableOpReadVariableOpoaettnoaty/kernel*
_output_shapes

: *
dtype0
v
oaettnoaty/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameoaettnoaty/bias
o
#oaettnoaty/bias/Read/ReadVariableOpReadVariableOpoaettnoaty/bias*
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
osutmzfngz/dsycfvoega/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameosutmzfngz/dsycfvoega/kernel

0osutmzfngz/dsycfvoega/kernel/Read/ReadVariableOpReadVariableOposutmzfngz/dsycfvoega/kernel*
_output_shapes
:	*
dtype0
©
&osutmzfngz/dsycfvoega/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&osutmzfngz/dsycfvoega/recurrent_kernel
¢
:osutmzfngz/dsycfvoega/recurrent_kernel/Read/ReadVariableOpReadVariableOp&osutmzfngz/dsycfvoega/recurrent_kernel*
_output_shapes
:	 *
dtype0

osutmzfngz/dsycfvoega/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameosutmzfngz/dsycfvoega/bias

.osutmzfngz/dsycfvoega/bias/Read/ReadVariableOpReadVariableOposutmzfngz/dsycfvoega/bias*
_output_shapes	
:*
dtype0
º
1osutmzfngz/dsycfvoega/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31osutmzfngz/dsycfvoega/input_gate_peephole_weights
³
Eosutmzfngz/dsycfvoega/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1osutmzfngz/dsycfvoega/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2osutmzfngz/dsycfvoega/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42osutmzfngz/dsycfvoega/forget_gate_peephole_weights
µ
Fosutmzfngz/dsycfvoega/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2osutmzfngz/dsycfvoega/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2osutmzfngz/dsycfvoega/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42osutmzfngz/dsycfvoega/output_gate_peephole_weights
µ
Fosutmzfngz/dsycfvoega/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2osutmzfngz/dsycfvoega/output_gate_peephole_weights*
_output_shapes
: *
dtype0

owshcilvwl/flzkvrshbq/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_nameowshcilvwl/flzkvrshbq/kernel

0owshcilvwl/flzkvrshbq/kernel/Read/ReadVariableOpReadVariableOpowshcilvwl/flzkvrshbq/kernel*
_output_shapes
:	 *
dtype0
©
&owshcilvwl/flzkvrshbq/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&owshcilvwl/flzkvrshbq/recurrent_kernel
¢
:owshcilvwl/flzkvrshbq/recurrent_kernel/Read/ReadVariableOpReadVariableOp&owshcilvwl/flzkvrshbq/recurrent_kernel*
_output_shapes
:	 *
dtype0

owshcilvwl/flzkvrshbq/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameowshcilvwl/flzkvrshbq/bias

.owshcilvwl/flzkvrshbq/bias/Read/ReadVariableOpReadVariableOpowshcilvwl/flzkvrshbq/bias*
_output_shapes	
:*
dtype0
º
1owshcilvwl/flzkvrshbq/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31owshcilvwl/flzkvrshbq/input_gate_peephole_weights
³
Eowshcilvwl/flzkvrshbq/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1owshcilvwl/flzkvrshbq/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2owshcilvwl/flzkvrshbq/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42owshcilvwl/flzkvrshbq/forget_gate_peephole_weights
µ
Fowshcilvwl/flzkvrshbq/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2owshcilvwl/flzkvrshbq/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2owshcilvwl/flzkvrshbq/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42owshcilvwl/flzkvrshbq/output_gate_peephole_weights
µ
Fowshcilvwl/flzkvrshbq/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2owshcilvwl/flzkvrshbq/output_gate_peephole_weights*
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
RMSprop/bpstkcuudk/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/bpstkcuudk/kernel/rms

1RMSprop/bpstkcuudk/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/bpstkcuudk/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/bpstkcuudk/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/bpstkcuudk/bias/rms

/RMSprop/bpstkcuudk/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/bpstkcuudk/bias/rms*
_output_shapes
:*
dtype0

RMSprop/oaettnoaty/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/oaettnoaty/kernel/rms

1RMSprop/oaettnoaty/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/oaettnoaty/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/oaettnoaty/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/oaettnoaty/bias/rms

/RMSprop/oaettnoaty/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/oaettnoaty/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/osutmzfngz/dsycfvoega/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/osutmzfngz/dsycfvoega/kernel/rms
¦
<RMSprop/osutmzfngz/dsycfvoega/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/osutmzfngz/dsycfvoega/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/osutmzfngz/dsycfvoega/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/osutmzfngz/dsycfvoega/recurrent_kernel/rms
º
FRMSprop/osutmzfngz/dsycfvoega/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/osutmzfngz/dsycfvoega/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/osutmzfngz/dsycfvoega/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/osutmzfngz/dsycfvoega/bias/rms

:RMSprop/osutmzfngz/dsycfvoega/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/osutmzfngz/dsycfvoega/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/osutmzfngz/dsycfvoega/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/osutmzfngz/dsycfvoega/input_gate_peephole_weights/rms
Ë
QRMSprop/osutmzfngz/dsycfvoega/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/osutmzfngz/dsycfvoega/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/osutmzfngz/dsycfvoega/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/osutmzfngz/dsycfvoega/forget_gate_peephole_weights/rms
Í
RRMSprop/osutmzfngz/dsycfvoega/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/osutmzfngz/dsycfvoega/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/osutmzfngz/dsycfvoega/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/osutmzfngz/dsycfvoega/output_gate_peephole_weights/rms
Í
RRMSprop/osutmzfngz/dsycfvoega/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/osutmzfngz/dsycfvoega/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/owshcilvwl/flzkvrshbq/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/owshcilvwl/flzkvrshbq/kernel/rms
¦
<RMSprop/owshcilvwl/flzkvrshbq/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/owshcilvwl/flzkvrshbq/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/owshcilvwl/flzkvrshbq/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/owshcilvwl/flzkvrshbq/recurrent_kernel/rms
º
FRMSprop/owshcilvwl/flzkvrshbq/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/owshcilvwl/flzkvrshbq/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/owshcilvwl/flzkvrshbq/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/owshcilvwl/flzkvrshbq/bias/rms

:RMSprop/owshcilvwl/flzkvrshbq/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/owshcilvwl/flzkvrshbq/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/owshcilvwl/flzkvrshbq/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/owshcilvwl/flzkvrshbq/input_gate_peephole_weights/rms
Ë
QRMSprop/owshcilvwl/flzkvrshbq/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/owshcilvwl/flzkvrshbq/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/owshcilvwl/flzkvrshbq/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/owshcilvwl/flzkvrshbq/forget_gate_peephole_weights/rms
Í
RRMSprop/owshcilvwl/flzkvrshbq/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/owshcilvwl/flzkvrshbq/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/owshcilvwl/flzkvrshbq/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/owshcilvwl/flzkvrshbq/output_gate_peephole_weights/rms
Í
RRMSprop/owshcilvwl/flzkvrshbq/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/owshcilvwl/flzkvrshbq/output_gate_peephole_weights/rms*
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
VARIABLE_VALUEbpstkcuudk/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbpstkcuudk/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEoaettnoaty/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEoaettnoaty/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEosutmzfngz/dsycfvoega/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&osutmzfngz/dsycfvoega/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEosutmzfngz/dsycfvoega/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1osutmzfngz/dsycfvoega/input_gate_peephole_weights&variables/5/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2osutmzfngz/dsycfvoega/forget_gate_peephole_weights&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2osutmzfngz/dsycfvoega/output_gate_peephole_weights&variables/7/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEowshcilvwl/flzkvrshbq/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&owshcilvwl/flzkvrshbq/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEowshcilvwl/flzkvrshbq/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1owshcilvwl/flzkvrshbq/input_gate_peephole_weights'variables/11/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2owshcilvwl/flzkvrshbq/forget_gate_peephole_weights'variables/12/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2owshcilvwl/flzkvrshbq/output_gate_peephole_weights'variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUERMSprop/bpstkcuudk/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/bpstkcuudk/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/oaettnoaty/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/oaettnoaty/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/osutmzfngz/dsycfvoega/kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/osutmzfngz/dsycfvoega/recurrent_kernel/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE&RMSprop/osutmzfngz/dsycfvoega/bias/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/osutmzfngz/dsycfvoega/input_gate_peephole_weights/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/osutmzfngz/dsycfvoega/forget_gate_peephole_weights/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/osutmzfngz/dsycfvoega/output_gate_peephole_weights/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/owshcilvwl/flzkvrshbq/kernel/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/owshcilvwl/flzkvrshbq/recurrent_kernel/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/owshcilvwl/flzkvrshbq/bias/rmsEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/owshcilvwl/flzkvrshbq/input_gate_peephole_weights/rmsEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/owshcilvwl/flzkvrshbq/forget_gate_peephole_weights/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/owshcilvwl/flzkvrshbq/output_gate_peephole_weights/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_jfowsgvbzwPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_jfowsgvbzwbpstkcuudk/kernelbpstkcuudk/biasosutmzfngz/dsycfvoega/kernel&osutmzfngz/dsycfvoega/recurrent_kernelosutmzfngz/dsycfvoega/bias1osutmzfngz/dsycfvoega/input_gate_peephole_weights2osutmzfngz/dsycfvoega/forget_gate_peephole_weights2osutmzfngz/dsycfvoega/output_gate_peephole_weightsowshcilvwl/flzkvrshbq/kernel&owshcilvwl/flzkvrshbq/recurrent_kernelowshcilvwl/flzkvrshbq/bias1owshcilvwl/flzkvrshbq/input_gate_peephole_weights2owshcilvwl/flzkvrshbq/forget_gate_peephole_weights2owshcilvwl/flzkvrshbq/output_gate_peephole_weightsoaettnoaty/kerneloaettnoaty/bias*
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
$__inference_signature_wrapper_656704
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ö
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%bpstkcuudk/kernel/Read/ReadVariableOp#bpstkcuudk/bias/Read/ReadVariableOp%oaettnoaty/kernel/Read/ReadVariableOp#oaettnoaty/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0osutmzfngz/dsycfvoega/kernel/Read/ReadVariableOp:osutmzfngz/dsycfvoega/recurrent_kernel/Read/ReadVariableOp.osutmzfngz/dsycfvoega/bias/Read/ReadVariableOpEosutmzfngz/dsycfvoega/input_gate_peephole_weights/Read/ReadVariableOpFosutmzfngz/dsycfvoega/forget_gate_peephole_weights/Read/ReadVariableOpFosutmzfngz/dsycfvoega/output_gate_peephole_weights/Read/ReadVariableOp0owshcilvwl/flzkvrshbq/kernel/Read/ReadVariableOp:owshcilvwl/flzkvrshbq/recurrent_kernel/Read/ReadVariableOp.owshcilvwl/flzkvrshbq/bias/Read/ReadVariableOpEowshcilvwl/flzkvrshbq/input_gate_peephole_weights/Read/ReadVariableOpFowshcilvwl/flzkvrshbq/forget_gate_peephole_weights/Read/ReadVariableOpFowshcilvwl/flzkvrshbq/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/bpstkcuudk/kernel/rms/Read/ReadVariableOp/RMSprop/bpstkcuudk/bias/rms/Read/ReadVariableOp1RMSprop/oaettnoaty/kernel/rms/Read/ReadVariableOp/RMSprop/oaettnoaty/bias/rms/Read/ReadVariableOp<RMSprop/osutmzfngz/dsycfvoega/kernel/rms/Read/ReadVariableOpFRMSprop/osutmzfngz/dsycfvoega/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/osutmzfngz/dsycfvoega/bias/rms/Read/ReadVariableOpQRMSprop/osutmzfngz/dsycfvoega/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/osutmzfngz/dsycfvoega/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/osutmzfngz/dsycfvoega/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/owshcilvwl/flzkvrshbq/kernel/rms/Read/ReadVariableOpFRMSprop/owshcilvwl/flzkvrshbq/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/owshcilvwl/flzkvrshbq/bias/rms/Read/ReadVariableOpQRMSprop/owshcilvwl/flzkvrshbq/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/owshcilvwl/flzkvrshbq/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/owshcilvwl/flzkvrshbq/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
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
__inference__traced_save_659653
å
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebpstkcuudk/kernelbpstkcuudk/biasoaettnoaty/kerneloaettnoaty/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhoosutmzfngz/dsycfvoega/kernel&osutmzfngz/dsycfvoega/recurrent_kernelosutmzfngz/dsycfvoega/bias1osutmzfngz/dsycfvoega/input_gate_peephole_weights2osutmzfngz/dsycfvoega/forget_gate_peephole_weights2osutmzfngz/dsycfvoega/output_gate_peephole_weightsowshcilvwl/flzkvrshbq/kernel&owshcilvwl/flzkvrshbq/recurrent_kernelowshcilvwl/flzkvrshbq/bias1owshcilvwl/flzkvrshbq/input_gate_peephole_weights2owshcilvwl/flzkvrshbq/forget_gate_peephole_weights2owshcilvwl/flzkvrshbq/output_gate_peephole_weightstotalcountRMSprop/bpstkcuudk/kernel/rmsRMSprop/bpstkcuudk/bias/rmsRMSprop/oaettnoaty/kernel/rmsRMSprop/oaettnoaty/bias/rms(RMSprop/osutmzfngz/dsycfvoega/kernel/rms2RMSprop/osutmzfngz/dsycfvoega/recurrent_kernel/rms&RMSprop/osutmzfngz/dsycfvoega/bias/rms=RMSprop/osutmzfngz/dsycfvoega/input_gate_peephole_weights/rms>RMSprop/osutmzfngz/dsycfvoega/forget_gate_peephole_weights/rms>RMSprop/osutmzfngz/dsycfvoega/output_gate_peephole_weights/rms(RMSprop/owshcilvwl/flzkvrshbq/kernel/rms2RMSprop/owshcilvwl/flzkvrshbq/recurrent_kernel/rms&RMSprop/owshcilvwl/flzkvrshbq/bias/rms=RMSprop/owshcilvwl/flzkvrshbq/input_gate_peephole_weights/rms>RMSprop/owshcilvwl/flzkvrshbq/forget_gate_peephole_weights/rms>RMSprop/owshcilvwl/flzkvrshbq/output_gate_peephole_weights/rms*3
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
"__inference__traced_restore_659780¥Û-
ßY

while_body_656293
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dsycfvoega_matmul_readvariableop_resource_0:	F
3while_dsycfvoega_matmul_1_readvariableop_resource_0:	 A
2while_dsycfvoega_biasadd_readvariableop_resource_0:	8
*while_dsycfvoega_readvariableop_resource_0: :
,while_dsycfvoega_readvariableop_1_resource_0: :
,while_dsycfvoega_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dsycfvoega_matmul_readvariableop_resource:	D
1while_dsycfvoega_matmul_1_readvariableop_resource:	 ?
0while_dsycfvoega_biasadd_readvariableop_resource:	6
(while_dsycfvoega_readvariableop_resource: 8
*while_dsycfvoega_readvariableop_1_resource: 8
*while_dsycfvoega_readvariableop_2_resource: ¢'while/dsycfvoega/BiasAdd/ReadVariableOp¢&while/dsycfvoega/MatMul/ReadVariableOp¢(while/dsycfvoega/MatMul_1/ReadVariableOp¢while/dsycfvoega/ReadVariableOp¢!while/dsycfvoega/ReadVariableOp_1¢!while/dsycfvoega/ReadVariableOp_2Ã
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
&while/dsycfvoega/MatMul/ReadVariableOpReadVariableOp1while_dsycfvoega_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dsycfvoega/MatMul/ReadVariableOpÑ
while/dsycfvoega/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMulÉ
(while/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp3while_dsycfvoega_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dsycfvoega/MatMul_1/ReadVariableOpº
while/dsycfvoega/MatMul_1MatMulwhile_placeholder_20while/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMul_1°
while/dsycfvoega/addAddV2!while/dsycfvoega/MatMul:product:0#while/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/addÂ
'while/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp2while_dsycfvoega_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dsycfvoega/BiasAdd/ReadVariableOp½
while/dsycfvoega/BiasAddBiasAddwhile/dsycfvoega/add:z:0/while/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/BiasAdd
 while/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dsycfvoega/split/split_dim
while/dsycfvoega/splitSplit)while/dsycfvoega/split/split_dim:output:0!while/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dsycfvoega/split©
while/dsycfvoega/ReadVariableOpReadVariableOp*while_dsycfvoega_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dsycfvoega/ReadVariableOp£
while/dsycfvoega/mulMul'while/dsycfvoega/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul¦
while/dsycfvoega/add_1AddV2while/dsycfvoega/split:output:0while/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_1
while/dsycfvoega/SigmoidSigmoidwhile/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid¯
!while/dsycfvoega/ReadVariableOp_1ReadVariableOp,while_dsycfvoega_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_1©
while/dsycfvoega/mul_1Mul)while/dsycfvoega/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_1¨
while/dsycfvoega/add_2AddV2while/dsycfvoega/split:output:1while/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_2
while/dsycfvoega/Sigmoid_1Sigmoidwhile/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_1
while/dsycfvoega/mul_2Mulwhile/dsycfvoega/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_2
while/dsycfvoega/TanhTanhwhile/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh¢
while/dsycfvoega/mul_3Mulwhile/dsycfvoega/Sigmoid:y:0while/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_3£
while/dsycfvoega/add_3AddV2while/dsycfvoega/mul_2:z:0while/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_3¯
!while/dsycfvoega/ReadVariableOp_2ReadVariableOp,while_dsycfvoega_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_2°
while/dsycfvoega/mul_4Mul)while/dsycfvoega/ReadVariableOp_2:value:0while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_4¨
while/dsycfvoega/add_4AddV2while/dsycfvoega/split:output:3while/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_4
while/dsycfvoega/Sigmoid_2Sigmoidwhile/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_2
while/dsycfvoega/Tanh_1Tanhwhile/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh_1¦
while/dsycfvoega/mul_5Mulwhile/dsycfvoega/Sigmoid_2:y:0while/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dsycfvoega/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dsycfvoega/mul_5:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dsycfvoega/add_3:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dsycfvoega_biasadd_readvariableop_resource2while_dsycfvoega_biasadd_readvariableop_resource_0"h
1while_dsycfvoega_matmul_1_readvariableop_resource3while_dsycfvoega_matmul_1_readvariableop_resource_0"d
/while_dsycfvoega_matmul_readvariableop_resource1while_dsycfvoega_matmul_readvariableop_resource_0"Z
*while_dsycfvoega_readvariableop_1_resource,while_dsycfvoega_readvariableop_1_resource_0"Z
*while_dsycfvoega_readvariableop_2_resource,while_dsycfvoega_readvariableop_2_resource_0"V
(while_dsycfvoega_readvariableop_resource*while_dsycfvoega_readvariableop_resource_0")
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
'while/dsycfvoega/BiasAdd/ReadVariableOp'while/dsycfvoega/BiasAdd/ReadVariableOp2P
&while/dsycfvoega/MatMul/ReadVariableOp&while/dsycfvoega/MatMul/ReadVariableOp2T
(while/dsycfvoega/MatMul_1/ReadVariableOp(while/dsycfvoega/MatMul_1/ReadVariableOp2B
while/dsycfvoega/ReadVariableOpwhile/dsycfvoega/ReadVariableOp2F
!while/dsycfvoega/ReadVariableOp_1!while/dsycfvoega/ReadVariableOp_12F
!while/dsycfvoega/ReadVariableOp_2!while/dsycfvoega/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
Ýh

F__inference_osutmzfngz_layer_call_and_return_conditional_losses_657898
inputs_0<
)dsycfvoega_matmul_readvariableop_resource:	>
+dsycfvoega_matmul_1_readvariableop_resource:	 9
*dsycfvoega_biasadd_readvariableop_resource:	0
"dsycfvoega_readvariableop_resource: 2
$dsycfvoega_readvariableop_1_resource: 2
$dsycfvoega_readvariableop_2_resource: 
identity¢!dsycfvoega/BiasAdd/ReadVariableOp¢ dsycfvoega/MatMul/ReadVariableOp¢"dsycfvoega/MatMul_1/ReadVariableOp¢dsycfvoega/ReadVariableOp¢dsycfvoega/ReadVariableOp_1¢dsycfvoega/ReadVariableOp_2¢whileF
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
 dsycfvoega/MatMul/ReadVariableOpReadVariableOp)dsycfvoega_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dsycfvoega/MatMul/ReadVariableOp§
dsycfvoega/MatMulMatMulstrided_slice_2:output:0(dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMulµ
"dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp+dsycfvoega_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dsycfvoega/MatMul_1/ReadVariableOp£
dsycfvoega/MatMul_1MatMulzeros:output:0*dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMul_1
dsycfvoega/addAddV2dsycfvoega/MatMul:product:0dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/add®
!dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp*dsycfvoega_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dsycfvoega/BiasAdd/ReadVariableOp¥
dsycfvoega/BiasAddBiasAdddsycfvoega/add:z:0)dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/BiasAddz
dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dsycfvoega/split/split_dimë
dsycfvoega/splitSplit#dsycfvoega/split/split_dim:output:0dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dsycfvoega/split
dsycfvoega/ReadVariableOpReadVariableOp"dsycfvoega_readvariableop_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp
dsycfvoega/mulMul!dsycfvoega/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul
dsycfvoega/add_1AddV2dsycfvoega/split:output:0dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_1{
dsycfvoega/SigmoidSigmoiddsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid
dsycfvoega/ReadVariableOp_1ReadVariableOp$dsycfvoega_readvariableop_1_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_1
dsycfvoega/mul_1Mul#dsycfvoega/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_1
dsycfvoega/add_2AddV2dsycfvoega/split:output:1dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_2
dsycfvoega/Sigmoid_1Sigmoiddsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_1
dsycfvoega/mul_2Muldsycfvoega/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_2w
dsycfvoega/TanhTanhdsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh
dsycfvoega/mul_3Muldsycfvoega/Sigmoid:y:0dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_3
dsycfvoega/add_3AddV2dsycfvoega/mul_2:z:0dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_3
dsycfvoega/ReadVariableOp_2ReadVariableOp$dsycfvoega_readvariableop_2_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_2
dsycfvoega/mul_4Mul#dsycfvoega/ReadVariableOp_2:value:0dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_4
dsycfvoega/add_4AddV2dsycfvoega/split:output:3dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_4
dsycfvoega/Sigmoid_2Sigmoiddsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_2v
dsycfvoega/Tanh_1Tanhdsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh_1
dsycfvoega/mul_5Muldsycfvoega/Sigmoid_2:y:0dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dsycfvoega_matmul_readvariableop_resource+dsycfvoega_matmul_1_readvariableop_resource*dsycfvoega_biasadd_readvariableop_resource"dsycfvoega_readvariableop_resource$dsycfvoega_readvariableop_1_resource$dsycfvoega_readvariableop_2_resource*
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
while_body_657797*
condR
while_cond_657796*Q
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
IdentityIdentitytranspose_1:y:0"^dsycfvoega/BiasAdd/ReadVariableOp!^dsycfvoega/MatMul/ReadVariableOp#^dsycfvoega/MatMul_1/ReadVariableOp^dsycfvoega/ReadVariableOp^dsycfvoega/ReadVariableOp_1^dsycfvoega/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dsycfvoega/BiasAdd/ReadVariableOp!dsycfvoega/BiasAdd/ReadVariableOp2D
 dsycfvoega/MatMul/ReadVariableOp dsycfvoega/MatMul/ReadVariableOp2H
"dsycfvoega/MatMul_1/ReadVariableOp"dsycfvoega/MatMul_1/ReadVariableOp26
dsycfvoega/ReadVariableOpdsycfvoega/ReadVariableOp2:
dsycfvoega/ReadVariableOp_1dsycfvoega/ReadVariableOp_12:
dsycfvoega/ReadVariableOp_2dsycfvoega/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

¯
F__inference_sequential_layer_call_and_return_conditional_losses_656659

jfowsgvbzw'
bpstkcuudk_656621:
bpstkcuudk_656623:$
osutmzfngz_656627:	$
osutmzfngz_656629:	  
osutmzfngz_656631:	
osutmzfngz_656633: 
osutmzfngz_656635: 
osutmzfngz_656637: $
owshcilvwl_656640:	 $
owshcilvwl_656642:	  
owshcilvwl_656644:	
owshcilvwl_656646: 
owshcilvwl_656648: 
owshcilvwl_656650: #
oaettnoaty_656653: 
oaettnoaty_656655:
identity¢"bpstkcuudk/StatefulPartitionedCall¢"oaettnoaty/StatefulPartitionedCall¢"osutmzfngz/StatefulPartitionedCall¢"owshcilvwl/StatefulPartitionedCall­
"bpstkcuudk/StatefulPartitionedCallStatefulPartitionedCall
jfowsgvbzwbpstkcuudk_656621bpstkcuudk_656623*
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
F__inference_bpstkcuudk_layer_call_and_return_conditional_losses_6555122$
"bpstkcuudk/StatefulPartitionedCall
xlcvyoxoxq/PartitionedCallPartitionedCall+bpstkcuudk/StatefulPartitionedCall:output:0*
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
F__inference_xlcvyoxoxq_layer_call_and_return_conditional_losses_6555312
xlcvyoxoxq/PartitionedCall
"osutmzfngz/StatefulPartitionedCallStatefulPartitionedCall#xlcvyoxoxq/PartitionedCall:output:0osutmzfngz_656627osutmzfngz_656629osutmzfngz_656631osutmzfngz_656633osutmzfngz_656635osutmzfngz_656637*
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_6563942$
"osutmzfngz/StatefulPartitionedCall
"owshcilvwl/StatefulPartitionedCallStatefulPartitionedCall+osutmzfngz/StatefulPartitionedCall:output:0owshcilvwl_656640owshcilvwl_656642owshcilvwl_656644owshcilvwl_656646owshcilvwl_656648owshcilvwl_656650*
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_6561802$
"owshcilvwl/StatefulPartitionedCallÆ
"oaettnoaty/StatefulPartitionedCallStatefulPartitionedCall+owshcilvwl/StatefulPartitionedCall:output:0oaettnoaty_656653oaettnoaty_656655*
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
F__inference_oaettnoaty_layer_call_and_return_conditional_losses_6559292$
"oaettnoaty/StatefulPartitionedCall
IdentityIdentity+oaettnoaty/StatefulPartitionedCall:output:0#^bpstkcuudk/StatefulPartitionedCall#^oaettnoaty/StatefulPartitionedCall#^osutmzfngz/StatefulPartitionedCall#^owshcilvwl/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"bpstkcuudk/StatefulPartitionedCall"bpstkcuudk/StatefulPartitionedCall2H
"oaettnoaty/StatefulPartitionedCall"oaettnoaty/StatefulPartitionedCall2H
"osutmzfngz/StatefulPartitionedCall"osutmzfngz/StatefulPartitionedCall2H
"owshcilvwl/StatefulPartitionedCall"owshcilvwl/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
jfowsgvbzw
	

+__inference_osutmzfngz_layer_call_fn_657667
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_6541392
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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_659469

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
Ç)
Å
while_body_655080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_flzkvrshbq_655104_0:	 ,
while_flzkvrshbq_655106_0:	 (
while_flzkvrshbq_655108_0:	'
while_flzkvrshbq_655110_0: '
while_flzkvrshbq_655112_0: '
while_flzkvrshbq_655114_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_flzkvrshbq_655104:	 *
while_flzkvrshbq_655106:	 &
while_flzkvrshbq_655108:	%
while_flzkvrshbq_655110: %
while_flzkvrshbq_655112: %
while_flzkvrshbq_655114: ¢(while/flzkvrshbq/StatefulPartitionedCallÃ
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
(while/flzkvrshbq/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_flzkvrshbq_655104_0while_flzkvrshbq_655106_0while_flzkvrshbq_655108_0while_flzkvrshbq_655110_0while_flzkvrshbq_655112_0while_flzkvrshbq_655114_0*
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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_6549842*
(while/flzkvrshbq/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/flzkvrshbq/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/flzkvrshbq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/flzkvrshbq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/flzkvrshbq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/flzkvrshbq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/flzkvrshbq/StatefulPartitionedCall:output:1)^while/flzkvrshbq/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/flzkvrshbq/StatefulPartitionedCall:output:2)^while/flzkvrshbq/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_flzkvrshbq_655104while_flzkvrshbq_655104_0"4
while_flzkvrshbq_655106while_flzkvrshbq_655106_0"4
while_flzkvrshbq_655108while_flzkvrshbq_655108_0"4
while_flzkvrshbq_655110while_flzkvrshbq_655110_0"4
while_flzkvrshbq_655112while_flzkvrshbq_655112_0"4
while_flzkvrshbq_655114while_flzkvrshbq_655114_0")
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
(while/flzkvrshbq/StatefulPartitionedCall(while/flzkvrshbq/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_sequential_layer_call_and_return_conditional_losses_656618

jfowsgvbzw'
bpstkcuudk_656580:
bpstkcuudk_656582:$
osutmzfngz_656586:	$
osutmzfngz_656588:	  
osutmzfngz_656590:	
osutmzfngz_656592: 
osutmzfngz_656594: 
osutmzfngz_656596: $
owshcilvwl_656599:	 $
owshcilvwl_656601:	  
owshcilvwl_656603:	
owshcilvwl_656605: 
owshcilvwl_656607: 
owshcilvwl_656609: #
oaettnoaty_656612: 
oaettnoaty_656614:
identity¢"bpstkcuudk/StatefulPartitionedCall¢"oaettnoaty/StatefulPartitionedCall¢"osutmzfngz/StatefulPartitionedCall¢"owshcilvwl/StatefulPartitionedCall­
"bpstkcuudk/StatefulPartitionedCallStatefulPartitionedCall
jfowsgvbzwbpstkcuudk_656580bpstkcuudk_656582*
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
F__inference_bpstkcuudk_layer_call_and_return_conditional_losses_6555122$
"bpstkcuudk/StatefulPartitionedCall
xlcvyoxoxq/PartitionedCallPartitionedCall+bpstkcuudk/StatefulPartitionedCall:output:0*
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
F__inference_xlcvyoxoxq_layer_call_and_return_conditional_losses_6555312
xlcvyoxoxq/PartitionedCall
"osutmzfngz/StatefulPartitionedCallStatefulPartitionedCall#xlcvyoxoxq/PartitionedCall:output:0osutmzfngz_656586osutmzfngz_656588osutmzfngz_656590osutmzfngz_656592osutmzfngz_656594osutmzfngz_656596*
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_6557122$
"osutmzfngz/StatefulPartitionedCall
"owshcilvwl/StatefulPartitionedCallStatefulPartitionedCall+osutmzfngz/StatefulPartitionedCall:output:0owshcilvwl_656599owshcilvwl_656601owshcilvwl_656603owshcilvwl_656605owshcilvwl_656607owshcilvwl_656609*
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_6559052$
"owshcilvwl/StatefulPartitionedCallÆ
"oaettnoaty/StatefulPartitionedCallStatefulPartitionedCall+owshcilvwl/StatefulPartitionedCall:output:0oaettnoaty_656612oaettnoaty_656614*
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
F__inference_oaettnoaty_layer_call_and_return_conditional_losses_6559292$
"oaettnoaty/StatefulPartitionedCall
IdentityIdentity+oaettnoaty/StatefulPartitionedCall:output:0#^bpstkcuudk/StatefulPartitionedCall#^oaettnoaty/StatefulPartitionedCall#^osutmzfngz/StatefulPartitionedCall#^owshcilvwl/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"bpstkcuudk/StatefulPartitionedCall"bpstkcuudk/StatefulPartitionedCall2H
"oaettnoaty/StatefulPartitionedCall"oaettnoaty/StatefulPartitionedCall2H
"osutmzfngz/StatefulPartitionedCall"osutmzfngz/StatefulPartitionedCall2H
"owshcilvwl/StatefulPartitionedCall"owshcilvwl/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
jfowsgvbzw
h

F__inference_osutmzfngz_layer_call_and_return_conditional_losses_655712

inputs<
)dsycfvoega_matmul_readvariableop_resource:	>
+dsycfvoega_matmul_1_readvariableop_resource:	 9
*dsycfvoega_biasadd_readvariableop_resource:	0
"dsycfvoega_readvariableop_resource: 2
$dsycfvoega_readvariableop_1_resource: 2
$dsycfvoega_readvariableop_2_resource: 
identity¢!dsycfvoega/BiasAdd/ReadVariableOp¢ dsycfvoega/MatMul/ReadVariableOp¢"dsycfvoega/MatMul_1/ReadVariableOp¢dsycfvoega/ReadVariableOp¢dsycfvoega/ReadVariableOp_1¢dsycfvoega/ReadVariableOp_2¢whileD
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
 dsycfvoega/MatMul/ReadVariableOpReadVariableOp)dsycfvoega_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dsycfvoega/MatMul/ReadVariableOp§
dsycfvoega/MatMulMatMulstrided_slice_2:output:0(dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMulµ
"dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp+dsycfvoega_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dsycfvoega/MatMul_1/ReadVariableOp£
dsycfvoega/MatMul_1MatMulzeros:output:0*dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMul_1
dsycfvoega/addAddV2dsycfvoega/MatMul:product:0dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/add®
!dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp*dsycfvoega_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dsycfvoega/BiasAdd/ReadVariableOp¥
dsycfvoega/BiasAddBiasAdddsycfvoega/add:z:0)dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/BiasAddz
dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dsycfvoega/split/split_dimë
dsycfvoega/splitSplit#dsycfvoega/split/split_dim:output:0dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dsycfvoega/split
dsycfvoega/ReadVariableOpReadVariableOp"dsycfvoega_readvariableop_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp
dsycfvoega/mulMul!dsycfvoega/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul
dsycfvoega/add_1AddV2dsycfvoega/split:output:0dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_1{
dsycfvoega/SigmoidSigmoiddsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid
dsycfvoega/ReadVariableOp_1ReadVariableOp$dsycfvoega_readvariableop_1_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_1
dsycfvoega/mul_1Mul#dsycfvoega/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_1
dsycfvoega/add_2AddV2dsycfvoega/split:output:1dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_2
dsycfvoega/Sigmoid_1Sigmoiddsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_1
dsycfvoega/mul_2Muldsycfvoega/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_2w
dsycfvoega/TanhTanhdsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh
dsycfvoega/mul_3Muldsycfvoega/Sigmoid:y:0dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_3
dsycfvoega/add_3AddV2dsycfvoega/mul_2:z:0dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_3
dsycfvoega/ReadVariableOp_2ReadVariableOp$dsycfvoega_readvariableop_2_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_2
dsycfvoega/mul_4Mul#dsycfvoega/ReadVariableOp_2:value:0dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_4
dsycfvoega/add_4AddV2dsycfvoega/split:output:3dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_4
dsycfvoega/Sigmoid_2Sigmoiddsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_2v
dsycfvoega/Tanh_1Tanhdsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh_1
dsycfvoega/mul_5Muldsycfvoega/Sigmoid_2:y:0dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dsycfvoega_matmul_readvariableop_resource+dsycfvoega_matmul_1_readvariableop_resource*dsycfvoega_biasadd_readvariableop_resource"dsycfvoega_readvariableop_resource$dsycfvoega_readvariableop_1_resource$dsycfvoega_readvariableop_2_resource*
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
while_body_655611*
condR
while_cond_655610*Q
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
IdentityIdentitytranspose_1:y:0"^dsycfvoega/BiasAdd/ReadVariableOp!^dsycfvoega/MatMul/ReadVariableOp#^dsycfvoega/MatMul_1/ReadVariableOp^dsycfvoega/ReadVariableOp^dsycfvoega/ReadVariableOp_1^dsycfvoega/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dsycfvoega/BiasAdd/ReadVariableOp!dsycfvoega/BiasAdd/ReadVariableOp2D
 dsycfvoega/MatMul/ReadVariableOp dsycfvoega/MatMul/ReadVariableOp2H
"dsycfvoega/MatMul_1/ReadVariableOp"dsycfvoega/MatMul_1/ReadVariableOp26
dsycfvoega/ReadVariableOpdsycfvoega/ReadVariableOp2:
dsycfvoega/ReadVariableOp_1dsycfvoega/ReadVariableOp_12:
dsycfvoega/ReadVariableOp_2dsycfvoega/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯

"__inference__traced_restore_659780
file_prefix8
"assignvariableop_bpstkcuudk_kernel:0
"assignvariableop_1_bpstkcuudk_bias:6
$assignvariableop_2_oaettnoaty_kernel: 0
"assignvariableop_3_oaettnoaty_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: B
/assignvariableop_9_osutmzfngz_dsycfvoega_kernel:	M
:assignvariableop_10_osutmzfngz_dsycfvoega_recurrent_kernel:	 =
.assignvariableop_11_osutmzfngz_dsycfvoega_bias:	S
Eassignvariableop_12_osutmzfngz_dsycfvoega_input_gate_peephole_weights: T
Fassignvariableop_13_osutmzfngz_dsycfvoega_forget_gate_peephole_weights: T
Fassignvariableop_14_osutmzfngz_dsycfvoega_output_gate_peephole_weights: C
0assignvariableop_15_owshcilvwl_flzkvrshbq_kernel:	 M
:assignvariableop_16_owshcilvwl_flzkvrshbq_recurrent_kernel:	 =
.assignvariableop_17_owshcilvwl_flzkvrshbq_bias:	S
Eassignvariableop_18_owshcilvwl_flzkvrshbq_input_gate_peephole_weights: T
Fassignvariableop_19_owshcilvwl_flzkvrshbq_forget_gate_peephole_weights: T
Fassignvariableop_20_owshcilvwl_flzkvrshbq_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: G
1assignvariableop_23_rmsprop_bpstkcuudk_kernel_rms:=
/assignvariableop_24_rmsprop_bpstkcuudk_bias_rms:C
1assignvariableop_25_rmsprop_oaettnoaty_kernel_rms: =
/assignvariableop_26_rmsprop_oaettnoaty_bias_rms:O
<assignvariableop_27_rmsprop_osutmzfngz_dsycfvoega_kernel_rms:	Y
Fassignvariableop_28_rmsprop_osutmzfngz_dsycfvoega_recurrent_kernel_rms:	 I
:assignvariableop_29_rmsprop_osutmzfngz_dsycfvoega_bias_rms:	_
Qassignvariableop_30_rmsprop_osutmzfngz_dsycfvoega_input_gate_peephole_weights_rms: `
Rassignvariableop_31_rmsprop_osutmzfngz_dsycfvoega_forget_gate_peephole_weights_rms: `
Rassignvariableop_32_rmsprop_osutmzfngz_dsycfvoega_output_gate_peephole_weights_rms: O
<assignvariableop_33_rmsprop_owshcilvwl_flzkvrshbq_kernel_rms:	 Y
Fassignvariableop_34_rmsprop_owshcilvwl_flzkvrshbq_recurrent_kernel_rms:	 I
:assignvariableop_35_rmsprop_owshcilvwl_flzkvrshbq_bias_rms:	_
Qassignvariableop_36_rmsprop_owshcilvwl_flzkvrshbq_input_gate_peephole_weights_rms: `
Rassignvariableop_37_rmsprop_owshcilvwl_flzkvrshbq_forget_gate_peephole_weights_rms: `
Rassignvariableop_38_rmsprop_owshcilvwl_flzkvrshbq_output_gate_peephole_weights_rms: 
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
AssignVariableOpAssignVariableOp"assignvariableop_bpstkcuudk_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_bpstkcuudk_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_oaettnoaty_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_oaettnoaty_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp/assignvariableop_9_osutmzfngz_dsycfvoega_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Â
AssignVariableOp_10AssignVariableOp:assignvariableop_10_osutmzfngz_dsycfvoega_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_osutmzfngz_dsycfvoega_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Í
AssignVariableOp_12AssignVariableOpEassignvariableop_12_osutmzfngz_dsycfvoega_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Î
AssignVariableOp_13AssignVariableOpFassignvariableop_13_osutmzfngz_dsycfvoega_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_osutmzfngz_dsycfvoega_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_owshcilvwl_flzkvrshbq_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Â
AssignVariableOp_16AssignVariableOp:assignvariableop_16_owshcilvwl_flzkvrshbq_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_owshcilvwl_flzkvrshbq_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Í
AssignVariableOp_18AssignVariableOpEassignvariableop_18_owshcilvwl_flzkvrshbq_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Î
AssignVariableOp_19AssignVariableOpFassignvariableop_19_owshcilvwl_flzkvrshbq_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Î
AssignVariableOp_20AssignVariableOpFassignvariableop_20_owshcilvwl_flzkvrshbq_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp1assignvariableop_23_rmsprop_bpstkcuudk_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_rmsprop_bpstkcuudk_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_oaettnoaty_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_oaettnoaty_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_osutmzfngz_dsycfvoega_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_rmsprop_osutmzfngz_dsycfvoega_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_rmsprop_osutmzfngz_dsycfvoega_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ù
AssignVariableOp_30AssignVariableOpQassignvariableop_30_rmsprop_osutmzfngz_dsycfvoega_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ú
AssignVariableOp_31AssignVariableOpRassignvariableop_31_rmsprop_osutmzfngz_dsycfvoega_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ú
AssignVariableOp_32AssignVariableOpRassignvariableop_32_rmsprop_osutmzfngz_dsycfvoega_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_rmsprop_owshcilvwl_flzkvrshbq_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_rmsprop_owshcilvwl_flzkvrshbq_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Â
AssignVariableOp_35AssignVariableOp:assignvariableop_35_rmsprop_owshcilvwl_flzkvrshbq_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ù
AssignVariableOp_36AssignVariableOpQassignvariableop_36_rmsprop_owshcilvwl_flzkvrshbq_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ú
AssignVariableOp_37AssignVariableOpRassignvariableop_37_rmsprop_owshcilvwl_flzkvrshbq_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ú
AssignVariableOp_38AssignVariableOpRassignvariableop_38_rmsprop_owshcilvwl_flzkvrshbq_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
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
ÿ
¿
+__inference_dsycfvoega_layer_call_fn_659268

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
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_6540392
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

b
F__inference_xlcvyoxoxq_layer_call_and_return_conditional_losses_657650

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


+__inference_sequential_layer_call_fn_656778

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
F__inference_sequential_layer_call_and_return_conditional_losses_6565052
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
Ñ

+__inference_owshcilvwl_layer_call_fn_658506

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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_6561802
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


+__inference_sequential_layer_call_fn_656741

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
F__inference_sequential_layer_call_and_return_conditional_losses_6559362
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
Z
Ê
__inference__traced_save_659653
file_prefix0
,savev2_bpstkcuudk_kernel_read_readvariableop.
*savev2_bpstkcuudk_bias_read_readvariableop0
,savev2_oaettnoaty_kernel_read_readvariableop.
*savev2_oaettnoaty_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_osutmzfngz_dsycfvoega_kernel_read_readvariableopE
Asavev2_osutmzfngz_dsycfvoega_recurrent_kernel_read_readvariableop9
5savev2_osutmzfngz_dsycfvoega_bias_read_readvariableopP
Lsavev2_osutmzfngz_dsycfvoega_input_gate_peephole_weights_read_readvariableopQ
Msavev2_osutmzfngz_dsycfvoega_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_osutmzfngz_dsycfvoega_output_gate_peephole_weights_read_readvariableop;
7savev2_owshcilvwl_flzkvrshbq_kernel_read_readvariableopE
Asavev2_owshcilvwl_flzkvrshbq_recurrent_kernel_read_readvariableop9
5savev2_owshcilvwl_flzkvrshbq_bias_read_readvariableopP
Lsavev2_owshcilvwl_flzkvrshbq_input_gate_peephole_weights_read_readvariableopQ
Msavev2_owshcilvwl_flzkvrshbq_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_owshcilvwl_flzkvrshbq_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_bpstkcuudk_kernel_rms_read_readvariableop:
6savev2_rmsprop_bpstkcuudk_bias_rms_read_readvariableop<
8savev2_rmsprop_oaettnoaty_kernel_rms_read_readvariableop:
6savev2_rmsprop_oaettnoaty_bias_rms_read_readvariableopG
Csavev2_rmsprop_osutmzfngz_dsycfvoega_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_osutmzfngz_dsycfvoega_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_osutmzfngz_dsycfvoega_bias_rms_read_readvariableop\
Xsavev2_rmsprop_osutmzfngz_dsycfvoega_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_osutmzfngz_dsycfvoega_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_osutmzfngz_dsycfvoega_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_owshcilvwl_flzkvrshbq_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_owshcilvwl_flzkvrshbq_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_owshcilvwl_flzkvrshbq_bias_rms_read_readvariableop\
Xsavev2_rmsprop_owshcilvwl_flzkvrshbq_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_owshcilvwl_flzkvrshbq_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_owshcilvwl_flzkvrshbq_output_gate_peephole_weights_rms_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_bpstkcuudk_kernel_read_readvariableop*savev2_bpstkcuudk_bias_read_readvariableop,savev2_oaettnoaty_kernel_read_readvariableop*savev2_oaettnoaty_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_osutmzfngz_dsycfvoega_kernel_read_readvariableopAsavev2_osutmzfngz_dsycfvoega_recurrent_kernel_read_readvariableop5savev2_osutmzfngz_dsycfvoega_bias_read_readvariableopLsavev2_osutmzfngz_dsycfvoega_input_gate_peephole_weights_read_readvariableopMsavev2_osutmzfngz_dsycfvoega_forget_gate_peephole_weights_read_readvariableopMsavev2_osutmzfngz_dsycfvoega_output_gate_peephole_weights_read_readvariableop7savev2_owshcilvwl_flzkvrshbq_kernel_read_readvariableopAsavev2_owshcilvwl_flzkvrshbq_recurrent_kernel_read_readvariableop5savev2_owshcilvwl_flzkvrshbq_bias_read_readvariableopLsavev2_owshcilvwl_flzkvrshbq_input_gate_peephole_weights_read_readvariableopMsavev2_owshcilvwl_flzkvrshbq_forget_gate_peephole_weights_read_readvariableopMsavev2_owshcilvwl_flzkvrshbq_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_bpstkcuudk_kernel_rms_read_readvariableop6savev2_rmsprop_bpstkcuudk_bias_rms_read_readvariableop8savev2_rmsprop_oaettnoaty_kernel_rms_read_readvariableop6savev2_rmsprop_oaettnoaty_bias_rms_read_readvariableopCsavev2_rmsprop_osutmzfngz_dsycfvoega_kernel_rms_read_readvariableopMsavev2_rmsprop_osutmzfngz_dsycfvoega_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_osutmzfngz_dsycfvoega_bias_rms_read_readvariableopXsavev2_rmsprop_osutmzfngz_dsycfvoega_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_osutmzfngz_dsycfvoega_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_osutmzfngz_dsycfvoega_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_owshcilvwl_flzkvrshbq_kernel_rms_read_readvariableopMsavev2_rmsprop_owshcilvwl_flzkvrshbq_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_owshcilvwl_flzkvrshbq_bias_rms_read_readvariableopXsavev2_rmsprop_owshcilvwl_flzkvrshbq_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_owshcilvwl_flzkvrshbq_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_owshcilvwl_flzkvrshbq_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
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
£h

F__inference_owshcilvwl_layer_call_and_return_conditional_losses_659226

inputs<
)flzkvrshbq_matmul_readvariableop_resource:	 >
+flzkvrshbq_matmul_1_readvariableop_resource:	 9
*flzkvrshbq_biasadd_readvariableop_resource:	0
"flzkvrshbq_readvariableop_resource: 2
$flzkvrshbq_readvariableop_1_resource: 2
$flzkvrshbq_readvariableop_2_resource: 
identity¢!flzkvrshbq/BiasAdd/ReadVariableOp¢ flzkvrshbq/MatMul/ReadVariableOp¢"flzkvrshbq/MatMul_1/ReadVariableOp¢flzkvrshbq/ReadVariableOp¢flzkvrshbq/ReadVariableOp_1¢flzkvrshbq/ReadVariableOp_2¢whileD
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
 flzkvrshbq/MatMul/ReadVariableOpReadVariableOp)flzkvrshbq_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 flzkvrshbq/MatMul/ReadVariableOp§
flzkvrshbq/MatMulMatMulstrided_slice_2:output:0(flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMulµ
"flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp+flzkvrshbq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"flzkvrshbq/MatMul_1/ReadVariableOp£
flzkvrshbq/MatMul_1MatMulzeros:output:0*flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMul_1
flzkvrshbq/addAddV2flzkvrshbq/MatMul:product:0flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/add®
!flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp*flzkvrshbq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!flzkvrshbq/BiasAdd/ReadVariableOp¥
flzkvrshbq/BiasAddBiasAddflzkvrshbq/add:z:0)flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/BiasAddz
flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
flzkvrshbq/split/split_dimë
flzkvrshbq/splitSplit#flzkvrshbq/split/split_dim:output:0flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
flzkvrshbq/split
flzkvrshbq/ReadVariableOpReadVariableOp"flzkvrshbq_readvariableop_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp
flzkvrshbq/mulMul!flzkvrshbq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul
flzkvrshbq/add_1AddV2flzkvrshbq/split:output:0flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_1{
flzkvrshbq/SigmoidSigmoidflzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid
flzkvrshbq/ReadVariableOp_1ReadVariableOp$flzkvrshbq_readvariableop_1_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_1
flzkvrshbq/mul_1Mul#flzkvrshbq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_1
flzkvrshbq/add_2AddV2flzkvrshbq/split:output:1flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_2
flzkvrshbq/Sigmoid_1Sigmoidflzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_1
flzkvrshbq/mul_2Mulflzkvrshbq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_2w
flzkvrshbq/TanhTanhflzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh
flzkvrshbq/mul_3Mulflzkvrshbq/Sigmoid:y:0flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_3
flzkvrshbq/add_3AddV2flzkvrshbq/mul_2:z:0flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_3
flzkvrshbq/ReadVariableOp_2ReadVariableOp$flzkvrshbq_readvariableop_2_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_2
flzkvrshbq/mul_4Mul#flzkvrshbq/ReadVariableOp_2:value:0flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_4
flzkvrshbq/add_4AddV2flzkvrshbq/split:output:3flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_4
flzkvrshbq/Sigmoid_2Sigmoidflzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_2v
flzkvrshbq/Tanh_1Tanhflzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh_1
flzkvrshbq/mul_5Mulflzkvrshbq/Sigmoid_2:y:0flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)flzkvrshbq_matmul_readvariableop_resource+flzkvrshbq_matmul_1_readvariableop_resource*flzkvrshbq_biasadd_readvariableop_resource"flzkvrshbq_readvariableop_resource$flzkvrshbq_readvariableop_1_resource$flzkvrshbq_readvariableop_2_resource*
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
while_body_659125*
condR
while_cond_659124*Q
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
IdentityIdentitystrided_slice_3:output:0"^flzkvrshbq/BiasAdd/ReadVariableOp!^flzkvrshbq/MatMul/ReadVariableOp#^flzkvrshbq/MatMul_1/ReadVariableOp^flzkvrshbq/ReadVariableOp^flzkvrshbq/ReadVariableOp_1^flzkvrshbq/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!flzkvrshbq/BiasAdd/ReadVariableOp!flzkvrshbq/BiasAdd/ReadVariableOp2D
 flzkvrshbq/MatMul/ReadVariableOp flzkvrshbq/MatMul/ReadVariableOp2H
"flzkvrshbq/MatMul_1/ReadVariableOp"flzkvrshbq/MatMul_1/ReadVariableOp26
flzkvrshbq/ReadVariableOpflzkvrshbq/ReadVariableOp2:
flzkvrshbq/ReadVariableOp_1flzkvrshbq/ReadVariableOp_12:
flzkvrshbq/ReadVariableOp_2flzkvrshbq/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
£h

F__inference_owshcilvwl_layer_call_and_return_conditional_losses_659046

inputs<
)flzkvrshbq_matmul_readvariableop_resource:	 >
+flzkvrshbq_matmul_1_readvariableop_resource:	 9
*flzkvrshbq_biasadd_readvariableop_resource:	0
"flzkvrshbq_readvariableop_resource: 2
$flzkvrshbq_readvariableop_1_resource: 2
$flzkvrshbq_readvariableop_2_resource: 
identity¢!flzkvrshbq/BiasAdd/ReadVariableOp¢ flzkvrshbq/MatMul/ReadVariableOp¢"flzkvrshbq/MatMul_1/ReadVariableOp¢flzkvrshbq/ReadVariableOp¢flzkvrshbq/ReadVariableOp_1¢flzkvrshbq/ReadVariableOp_2¢whileD
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
 flzkvrshbq/MatMul/ReadVariableOpReadVariableOp)flzkvrshbq_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 flzkvrshbq/MatMul/ReadVariableOp§
flzkvrshbq/MatMulMatMulstrided_slice_2:output:0(flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMulµ
"flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp+flzkvrshbq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"flzkvrshbq/MatMul_1/ReadVariableOp£
flzkvrshbq/MatMul_1MatMulzeros:output:0*flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMul_1
flzkvrshbq/addAddV2flzkvrshbq/MatMul:product:0flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/add®
!flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp*flzkvrshbq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!flzkvrshbq/BiasAdd/ReadVariableOp¥
flzkvrshbq/BiasAddBiasAddflzkvrshbq/add:z:0)flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/BiasAddz
flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
flzkvrshbq/split/split_dimë
flzkvrshbq/splitSplit#flzkvrshbq/split/split_dim:output:0flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
flzkvrshbq/split
flzkvrshbq/ReadVariableOpReadVariableOp"flzkvrshbq_readvariableop_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp
flzkvrshbq/mulMul!flzkvrshbq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul
flzkvrshbq/add_1AddV2flzkvrshbq/split:output:0flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_1{
flzkvrshbq/SigmoidSigmoidflzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid
flzkvrshbq/ReadVariableOp_1ReadVariableOp$flzkvrshbq_readvariableop_1_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_1
flzkvrshbq/mul_1Mul#flzkvrshbq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_1
flzkvrshbq/add_2AddV2flzkvrshbq/split:output:1flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_2
flzkvrshbq/Sigmoid_1Sigmoidflzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_1
flzkvrshbq/mul_2Mulflzkvrshbq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_2w
flzkvrshbq/TanhTanhflzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh
flzkvrshbq/mul_3Mulflzkvrshbq/Sigmoid:y:0flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_3
flzkvrshbq/add_3AddV2flzkvrshbq/mul_2:z:0flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_3
flzkvrshbq/ReadVariableOp_2ReadVariableOp$flzkvrshbq_readvariableop_2_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_2
flzkvrshbq/mul_4Mul#flzkvrshbq/ReadVariableOp_2:value:0flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_4
flzkvrshbq/add_4AddV2flzkvrshbq/split:output:3flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_4
flzkvrshbq/Sigmoid_2Sigmoidflzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_2v
flzkvrshbq/Tanh_1Tanhflzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh_1
flzkvrshbq/mul_5Mulflzkvrshbq/Sigmoid_2:y:0flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)flzkvrshbq_matmul_readvariableop_resource+flzkvrshbq_matmul_1_readvariableop_resource*flzkvrshbq_biasadd_readvariableop_resource"flzkvrshbq_readvariableop_resource$flzkvrshbq_readvariableop_1_resource$flzkvrshbq_readvariableop_2_resource*
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
while_body_658945*
condR
while_cond_658944*Q
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
IdentityIdentitystrided_slice_3:output:0"^flzkvrshbq/BiasAdd/ReadVariableOp!^flzkvrshbq/MatMul/ReadVariableOp#^flzkvrshbq/MatMul_1/ReadVariableOp^flzkvrshbq/ReadVariableOp^flzkvrshbq/ReadVariableOp_1^flzkvrshbq/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!flzkvrshbq/BiasAdd/ReadVariableOp!flzkvrshbq/BiasAdd/ReadVariableOp2D
 flzkvrshbq/MatMul/ReadVariableOp flzkvrshbq/MatMul/ReadVariableOp2H
"flzkvrshbq/MatMul_1/ReadVariableOp"flzkvrshbq/MatMul_1/ReadVariableOp26
flzkvrshbq/ReadVariableOpflzkvrshbq/ReadVariableOp2:
flzkvrshbq/ReadVariableOp_1flzkvrshbq/ReadVariableOp_12:
flzkvrshbq/ReadVariableOp_2flzkvrshbq/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
F
ã
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_654402

inputs$
dsycfvoega_654303:	$
dsycfvoega_654305:	  
dsycfvoega_654307:	
dsycfvoega_654309: 
dsycfvoega_654311: 
dsycfvoega_654313: 
identity¢"dsycfvoega/StatefulPartitionedCall¢whileD
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
"dsycfvoega/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0dsycfvoega_654303dsycfvoega_654305dsycfvoega_654307dsycfvoega_654309dsycfvoega_654311dsycfvoega_654313*
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
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_6542262$
"dsycfvoega/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0dsycfvoega_654303dsycfvoega_654305dsycfvoega_654307dsycfvoega_654309dsycfvoega_654311dsycfvoega_654313*
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
while_body_654322*
condR
while_cond_654321*Q
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
IdentityIdentitytranspose_1:y:0#^dsycfvoega/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"dsycfvoega/StatefulPartitionedCall"dsycfvoega/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò	
÷
F__inference_oaettnoaty_layer_call_and_return_conditional_losses_655929

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


å
while_cond_659124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_659124___redundant_placeholder04
0while_while_cond_659124___redundant_placeholder14
0while_while_cond_659124___redundant_placeholder24
0while_while_cond_659124___redundant_placeholder34
0while_while_cond_659124___redundant_placeholder44
0while_while_cond_659124___redundant_placeholder54
0while_while_cond_659124___redundant_placeholder6
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
owshcilvwl_while_body_6570752
.owshcilvwl_while_owshcilvwl_while_loop_counter8
4owshcilvwl_while_owshcilvwl_while_maximum_iterations 
owshcilvwl_while_placeholder"
owshcilvwl_while_placeholder_1"
owshcilvwl_while_placeholder_2"
owshcilvwl_while_placeholder_31
-owshcilvwl_while_owshcilvwl_strided_slice_1_0m
iowshcilvwl_while_tensorarrayv2read_tensorlistgetitem_owshcilvwl_tensorarrayunstack_tensorlistfromtensor_0O
<owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource_0:	 Q
>owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource_0:	 L
=owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource_0:	C
5owshcilvwl_while_flzkvrshbq_readvariableop_resource_0: E
7owshcilvwl_while_flzkvrshbq_readvariableop_1_resource_0: E
7owshcilvwl_while_flzkvrshbq_readvariableop_2_resource_0: 
owshcilvwl_while_identity
owshcilvwl_while_identity_1
owshcilvwl_while_identity_2
owshcilvwl_while_identity_3
owshcilvwl_while_identity_4
owshcilvwl_while_identity_5/
+owshcilvwl_while_owshcilvwl_strided_slice_1k
gowshcilvwl_while_tensorarrayv2read_tensorlistgetitem_owshcilvwl_tensorarrayunstack_tensorlistfromtensorM
:owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource:	 O
<owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource:	 J
;owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource:	A
3owshcilvwl_while_flzkvrshbq_readvariableop_resource: C
5owshcilvwl_while_flzkvrshbq_readvariableop_1_resource: C
5owshcilvwl_while_flzkvrshbq_readvariableop_2_resource: ¢2owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp¢1owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp¢3owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp¢*owshcilvwl/while/flzkvrshbq/ReadVariableOp¢,owshcilvwl/while/flzkvrshbq/ReadVariableOp_1¢,owshcilvwl/while/flzkvrshbq/ReadVariableOp_2Ù
Bowshcilvwl/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bowshcilvwl/while/TensorArrayV2Read/TensorListGetItem/element_shape
4owshcilvwl/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiowshcilvwl_while_tensorarrayv2read_tensorlistgetitem_owshcilvwl_tensorarrayunstack_tensorlistfromtensor_0owshcilvwl_while_placeholderKowshcilvwl/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4owshcilvwl/while/TensorArrayV2Read/TensorListGetItemä
1owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp<owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOpý
"owshcilvwl/while/flzkvrshbq/MatMulMatMul;owshcilvwl/while/TensorArrayV2Read/TensorListGetItem:item:09owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"owshcilvwl/while/flzkvrshbq/MatMulê
3owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp>owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOpæ
$owshcilvwl/while/flzkvrshbq/MatMul_1MatMulowshcilvwl_while_placeholder_2;owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$owshcilvwl/while/flzkvrshbq/MatMul_1Ü
owshcilvwl/while/flzkvrshbq/addAddV2,owshcilvwl/while/flzkvrshbq/MatMul:product:0.owshcilvwl/while/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
owshcilvwl/while/flzkvrshbq/addã
2owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp=owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOpé
#owshcilvwl/while/flzkvrshbq/BiasAddBiasAdd#owshcilvwl/while/flzkvrshbq/add:z:0:owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#owshcilvwl/while/flzkvrshbq/BiasAdd
+owshcilvwl/while/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+owshcilvwl/while/flzkvrshbq/split/split_dim¯
!owshcilvwl/while/flzkvrshbq/splitSplit4owshcilvwl/while/flzkvrshbq/split/split_dim:output:0,owshcilvwl/while/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!owshcilvwl/while/flzkvrshbq/splitÊ
*owshcilvwl/while/flzkvrshbq/ReadVariableOpReadVariableOp5owshcilvwl_while_flzkvrshbq_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*owshcilvwl/while/flzkvrshbq/ReadVariableOpÏ
owshcilvwl/while/flzkvrshbq/mulMul2owshcilvwl/while/flzkvrshbq/ReadVariableOp:value:0owshcilvwl_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
owshcilvwl/while/flzkvrshbq/mulÒ
!owshcilvwl/while/flzkvrshbq/add_1AddV2*owshcilvwl/while/flzkvrshbq/split:output:0#owshcilvwl/while/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/add_1®
#owshcilvwl/while/flzkvrshbq/SigmoidSigmoid%owshcilvwl/while/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#owshcilvwl/while/flzkvrshbq/SigmoidÐ
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_1ReadVariableOp7owshcilvwl_while_flzkvrshbq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_1Õ
!owshcilvwl/while/flzkvrshbq/mul_1Mul4owshcilvwl/while/flzkvrshbq/ReadVariableOp_1:value:0owshcilvwl_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/mul_1Ô
!owshcilvwl/while/flzkvrshbq/add_2AddV2*owshcilvwl/while/flzkvrshbq/split:output:1%owshcilvwl/while/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/add_2²
%owshcilvwl/while/flzkvrshbq/Sigmoid_1Sigmoid%owshcilvwl/while/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%owshcilvwl/while/flzkvrshbq/Sigmoid_1Ê
!owshcilvwl/while/flzkvrshbq/mul_2Mul)owshcilvwl/while/flzkvrshbq/Sigmoid_1:y:0owshcilvwl_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/mul_2ª
 owshcilvwl/while/flzkvrshbq/TanhTanh*owshcilvwl/while/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 owshcilvwl/while/flzkvrshbq/TanhÎ
!owshcilvwl/while/flzkvrshbq/mul_3Mul'owshcilvwl/while/flzkvrshbq/Sigmoid:y:0$owshcilvwl/while/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/mul_3Ï
!owshcilvwl/while/flzkvrshbq/add_3AddV2%owshcilvwl/while/flzkvrshbq/mul_2:z:0%owshcilvwl/while/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/add_3Ð
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_2ReadVariableOp7owshcilvwl_while_flzkvrshbq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_2Ü
!owshcilvwl/while/flzkvrshbq/mul_4Mul4owshcilvwl/while/flzkvrshbq/ReadVariableOp_2:value:0%owshcilvwl/while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/mul_4Ô
!owshcilvwl/while/flzkvrshbq/add_4AddV2*owshcilvwl/while/flzkvrshbq/split:output:3%owshcilvwl/while/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/add_4²
%owshcilvwl/while/flzkvrshbq/Sigmoid_2Sigmoid%owshcilvwl/while/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%owshcilvwl/while/flzkvrshbq/Sigmoid_2©
"owshcilvwl/while/flzkvrshbq/Tanh_1Tanh%owshcilvwl/while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"owshcilvwl/while/flzkvrshbq/Tanh_1Ò
!owshcilvwl/while/flzkvrshbq/mul_5Mul)owshcilvwl/while/flzkvrshbq/Sigmoid_2:y:0&owshcilvwl/while/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/mul_5
5owshcilvwl/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemowshcilvwl_while_placeholder_1owshcilvwl_while_placeholder%owshcilvwl/while/flzkvrshbq/mul_5:z:0*
_output_shapes
: *
element_dtype027
5owshcilvwl/while/TensorArrayV2Write/TensorListSetItemr
owshcilvwl/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
owshcilvwl/while/add/y
owshcilvwl/while/addAddV2owshcilvwl_while_placeholderowshcilvwl/while/add/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/while/addv
owshcilvwl/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
owshcilvwl/while/add_1/y­
owshcilvwl/while/add_1AddV2.owshcilvwl_while_owshcilvwl_while_loop_counter!owshcilvwl/while/add_1/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/while/add_1©
owshcilvwl/while/IdentityIdentityowshcilvwl/while/add_1:z:03^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
owshcilvwl/while/IdentityÇ
owshcilvwl/while/Identity_1Identity4owshcilvwl_while_owshcilvwl_while_maximum_iterations3^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
owshcilvwl/while/Identity_1«
owshcilvwl/while/Identity_2Identityowshcilvwl/while/add:z:03^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
owshcilvwl/while/Identity_2Ø
owshcilvwl/while/Identity_3IdentityEowshcilvwl/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
owshcilvwl/while/Identity_3É
owshcilvwl/while/Identity_4Identity%owshcilvwl/while/flzkvrshbq/mul_5:z:03^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/while/Identity_4É
owshcilvwl/while/Identity_5Identity%owshcilvwl/while/flzkvrshbq/add_3:z:03^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/while/Identity_5"|
;owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource=owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource_0"~
<owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource>owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource_0"z
:owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource<owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource_0"p
5owshcilvwl_while_flzkvrshbq_readvariableop_1_resource7owshcilvwl_while_flzkvrshbq_readvariableop_1_resource_0"p
5owshcilvwl_while_flzkvrshbq_readvariableop_2_resource7owshcilvwl_while_flzkvrshbq_readvariableop_2_resource_0"l
3owshcilvwl_while_flzkvrshbq_readvariableop_resource5owshcilvwl_while_flzkvrshbq_readvariableop_resource_0"?
owshcilvwl_while_identity"owshcilvwl/while/Identity:output:0"C
owshcilvwl_while_identity_1$owshcilvwl/while/Identity_1:output:0"C
owshcilvwl_while_identity_2$owshcilvwl/while/Identity_2:output:0"C
owshcilvwl_while_identity_3$owshcilvwl/while/Identity_3:output:0"C
owshcilvwl_while_identity_4$owshcilvwl/while/Identity_4:output:0"C
owshcilvwl_while_identity_5$owshcilvwl/while/Identity_5:output:0"\
+owshcilvwl_while_owshcilvwl_strided_slice_1-owshcilvwl_while_owshcilvwl_strided_slice_1_0"Ô
gowshcilvwl_while_tensorarrayv2read_tensorlistgetitem_owshcilvwl_tensorarrayunstack_tensorlistfromtensoriowshcilvwl_while_tensorarrayv2read_tensorlistgetitem_owshcilvwl_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2f
1owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp1owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp2j
3owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp3owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp2X
*owshcilvwl/while/flzkvrshbq/ReadVariableOp*owshcilvwl/while/flzkvrshbq/ReadVariableOp2\
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_1,owshcilvwl/while/flzkvrshbq/ReadVariableOp_12\
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_2,owshcilvwl/while/flzkvrshbq/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
owshcilvwl_while_cond_6574782
.owshcilvwl_while_owshcilvwl_while_loop_counter8
4owshcilvwl_while_owshcilvwl_while_maximum_iterations 
owshcilvwl_while_placeholder"
owshcilvwl_while_placeholder_1"
owshcilvwl_while_placeholder_2"
owshcilvwl_while_placeholder_34
0owshcilvwl_while_less_owshcilvwl_strided_slice_1J
Fowshcilvwl_while_owshcilvwl_while_cond_657478___redundant_placeholder0J
Fowshcilvwl_while_owshcilvwl_while_cond_657478___redundant_placeholder1J
Fowshcilvwl_while_owshcilvwl_while_cond_657478___redundant_placeholder2J
Fowshcilvwl_while_owshcilvwl_while_cond_657478___redundant_placeholder3J
Fowshcilvwl_while_owshcilvwl_while_cond_657478___redundant_placeholder4J
Fowshcilvwl_while_owshcilvwl_while_cond_657478___redundant_placeholder5J
Fowshcilvwl_while_owshcilvwl_while_cond_657478___redundant_placeholder6
owshcilvwl_while_identity
§
owshcilvwl/while/LessLessowshcilvwl_while_placeholder0owshcilvwl_while_less_owshcilvwl_strided_slice_1*
T0*
_output_shapes
: 2
owshcilvwl/while/Less~
owshcilvwl/while/IdentityIdentityowshcilvwl/while/Less:z:0*
T0
*
_output_shapes
: 2
owshcilvwl/while/Identity"?
owshcilvwl_while_identity"owshcilvwl/while/Identity:output:0*(
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
Û

'sequential_owshcilvwl_while_body_653845H
Dsequential_owshcilvwl_while_sequential_owshcilvwl_while_loop_counterN
Jsequential_owshcilvwl_while_sequential_owshcilvwl_while_maximum_iterations+
'sequential_owshcilvwl_while_placeholder-
)sequential_owshcilvwl_while_placeholder_1-
)sequential_owshcilvwl_while_placeholder_2-
)sequential_owshcilvwl_while_placeholder_3G
Csequential_owshcilvwl_while_sequential_owshcilvwl_strided_slice_1_0
sequential_owshcilvwl_while_tensorarrayv2read_tensorlistgetitem_sequential_owshcilvwl_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource_0:	 \
Isequential_owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource_0:	 W
Hsequential_owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource_0:	N
@sequential_owshcilvwl_while_flzkvrshbq_readvariableop_resource_0: P
Bsequential_owshcilvwl_while_flzkvrshbq_readvariableop_1_resource_0: P
Bsequential_owshcilvwl_while_flzkvrshbq_readvariableop_2_resource_0: (
$sequential_owshcilvwl_while_identity*
&sequential_owshcilvwl_while_identity_1*
&sequential_owshcilvwl_while_identity_2*
&sequential_owshcilvwl_while_identity_3*
&sequential_owshcilvwl_while_identity_4*
&sequential_owshcilvwl_while_identity_5E
Asequential_owshcilvwl_while_sequential_owshcilvwl_strided_slice_1
}sequential_owshcilvwl_while_tensorarrayv2read_tensorlistgetitem_sequential_owshcilvwl_tensorarrayunstack_tensorlistfromtensorX
Esequential_owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource:	 Z
Gsequential_owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource:	 U
Fsequential_owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource:	L
>sequential_owshcilvwl_while_flzkvrshbq_readvariableop_resource: N
@sequential_owshcilvwl_while_flzkvrshbq_readvariableop_1_resource: N
@sequential_owshcilvwl_while_flzkvrshbq_readvariableop_2_resource: ¢=sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp¢<sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp¢>sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp¢5sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp¢7sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_1¢7sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2ï
Msequential/owshcilvwl/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/owshcilvwl/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/owshcilvwl/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_owshcilvwl_while_tensorarrayv2read_tensorlistgetitem_sequential_owshcilvwl_tensorarrayunstack_tensorlistfromtensor_0'sequential_owshcilvwl_while_placeholderVsequential/owshcilvwl/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/owshcilvwl/while/TensorArrayV2Read/TensorListGetItem
<sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOpReadVariableOpGsequential_owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp©
-sequential/owshcilvwl/while/flzkvrshbq/MatMulMatMulFsequential/owshcilvwl/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/owshcilvwl/while/flzkvrshbq/MatMul
>sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOpIsequential_owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp
/sequential/owshcilvwl/while/flzkvrshbq/MatMul_1MatMul)sequential_owshcilvwl_while_placeholder_2Fsequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/owshcilvwl/while/flzkvrshbq/MatMul_1
*sequential/owshcilvwl/while/flzkvrshbq/addAddV27sequential/owshcilvwl/while/flzkvrshbq/MatMul:product:09sequential/owshcilvwl/while/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/owshcilvwl/while/flzkvrshbq/add
=sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOpHsequential_owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp
.sequential/owshcilvwl/while/flzkvrshbq/BiasAddBiasAdd.sequential/owshcilvwl/while/flzkvrshbq/add:z:0Esequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/owshcilvwl/while/flzkvrshbq/BiasAdd²
6sequential/owshcilvwl/while/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/owshcilvwl/while/flzkvrshbq/split/split_dimÛ
,sequential/owshcilvwl/while/flzkvrshbq/splitSplit?sequential/owshcilvwl/while/flzkvrshbq/split/split_dim:output:07sequential/owshcilvwl/while/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/owshcilvwl/while/flzkvrshbq/splitë
5sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOpReadVariableOp@sequential_owshcilvwl_while_flzkvrshbq_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOpû
*sequential/owshcilvwl/while/flzkvrshbq/mulMul=sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp:value:0)sequential_owshcilvwl_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/owshcilvwl/while/flzkvrshbq/mulþ
,sequential/owshcilvwl/while/flzkvrshbq/add_1AddV25sequential/owshcilvwl/while/flzkvrshbq/split:output:0.sequential/owshcilvwl/while/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/owshcilvwl/while/flzkvrshbq/add_1Ï
.sequential/owshcilvwl/while/flzkvrshbq/SigmoidSigmoid0sequential/owshcilvwl/while/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/owshcilvwl/while/flzkvrshbq/Sigmoidñ
7sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_1ReadVariableOpBsequential_owshcilvwl_while_flzkvrshbq_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_1
,sequential/owshcilvwl/while/flzkvrshbq/mul_1Mul?sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_1:value:0)sequential_owshcilvwl_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/owshcilvwl/while/flzkvrshbq/mul_1
,sequential/owshcilvwl/while/flzkvrshbq/add_2AddV25sequential/owshcilvwl/while/flzkvrshbq/split:output:10sequential/owshcilvwl/while/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/owshcilvwl/while/flzkvrshbq/add_2Ó
0sequential/owshcilvwl/while/flzkvrshbq/Sigmoid_1Sigmoid0sequential/owshcilvwl/while/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/owshcilvwl/while/flzkvrshbq/Sigmoid_1ö
,sequential/owshcilvwl/while/flzkvrshbq/mul_2Mul4sequential/owshcilvwl/while/flzkvrshbq/Sigmoid_1:y:0)sequential_owshcilvwl_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/owshcilvwl/while/flzkvrshbq/mul_2Ë
+sequential/owshcilvwl/while/flzkvrshbq/TanhTanh5sequential/owshcilvwl/while/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/owshcilvwl/while/flzkvrshbq/Tanhú
,sequential/owshcilvwl/while/flzkvrshbq/mul_3Mul2sequential/owshcilvwl/while/flzkvrshbq/Sigmoid:y:0/sequential/owshcilvwl/while/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/owshcilvwl/while/flzkvrshbq/mul_3û
,sequential/owshcilvwl/while/flzkvrshbq/add_3AddV20sequential/owshcilvwl/while/flzkvrshbq/mul_2:z:00sequential/owshcilvwl/while/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/owshcilvwl/while/flzkvrshbq/add_3ñ
7sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2ReadVariableOpBsequential_owshcilvwl_while_flzkvrshbq_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2
,sequential/owshcilvwl/while/flzkvrshbq/mul_4Mul?sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2:value:00sequential/owshcilvwl/while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/owshcilvwl/while/flzkvrshbq/mul_4
,sequential/owshcilvwl/while/flzkvrshbq/add_4AddV25sequential/owshcilvwl/while/flzkvrshbq/split:output:30sequential/owshcilvwl/while/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/owshcilvwl/while/flzkvrshbq/add_4Ó
0sequential/owshcilvwl/while/flzkvrshbq/Sigmoid_2Sigmoid0sequential/owshcilvwl/while/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/owshcilvwl/while/flzkvrshbq/Sigmoid_2Ê
-sequential/owshcilvwl/while/flzkvrshbq/Tanh_1Tanh0sequential/owshcilvwl/while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/owshcilvwl/while/flzkvrshbq/Tanh_1þ
,sequential/owshcilvwl/while/flzkvrshbq/mul_5Mul4sequential/owshcilvwl/while/flzkvrshbq/Sigmoid_2:y:01sequential/owshcilvwl/while/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/owshcilvwl/while/flzkvrshbq/mul_5Ì
@sequential/owshcilvwl/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_owshcilvwl_while_placeholder_1'sequential_owshcilvwl_while_placeholder0sequential/owshcilvwl/while/flzkvrshbq/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/owshcilvwl/while/TensorArrayV2Write/TensorListSetItem
!sequential/owshcilvwl/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/owshcilvwl/while/add/yÁ
sequential/owshcilvwl/while/addAddV2'sequential_owshcilvwl_while_placeholder*sequential/owshcilvwl/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/owshcilvwl/while/add
#sequential/owshcilvwl/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/owshcilvwl/while/add_1/yä
!sequential/owshcilvwl/while/add_1AddV2Dsequential_owshcilvwl_while_sequential_owshcilvwl_while_loop_counter,sequential/owshcilvwl/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/owshcilvwl/while/add_1
$sequential/owshcilvwl/while/IdentityIdentity%sequential/owshcilvwl/while/add_1:z:0>^sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp=^sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp?^sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp6^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp8^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_18^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/owshcilvwl/while/Identityµ
&sequential/owshcilvwl/while/Identity_1IdentityJsequential_owshcilvwl_while_sequential_owshcilvwl_while_maximum_iterations>^sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp=^sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp?^sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp6^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp8^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_18^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/owshcilvwl/while/Identity_1
&sequential/owshcilvwl/while/Identity_2Identity#sequential/owshcilvwl/while/add:z:0>^sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp=^sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp?^sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp6^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp8^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_18^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/owshcilvwl/while/Identity_2»
&sequential/owshcilvwl/while/Identity_3IdentityPsequential/owshcilvwl/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp=^sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp?^sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp6^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp8^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_18^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/owshcilvwl/while/Identity_3¬
&sequential/owshcilvwl/while/Identity_4Identity0sequential/owshcilvwl/while/flzkvrshbq/mul_5:z:0>^sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp=^sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp?^sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp6^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp8^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_18^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/while/Identity_4¬
&sequential/owshcilvwl/while/Identity_5Identity0sequential/owshcilvwl/while/flzkvrshbq/add_3:z:0>^sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp=^sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp?^sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp6^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp8^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_18^sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/while/Identity_5"
Fsequential_owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resourceHsequential_owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource_0"
Gsequential_owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resourceIsequential_owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource_0"
Esequential_owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resourceGsequential_owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource_0"
@sequential_owshcilvwl_while_flzkvrshbq_readvariableop_1_resourceBsequential_owshcilvwl_while_flzkvrshbq_readvariableop_1_resource_0"
@sequential_owshcilvwl_while_flzkvrshbq_readvariableop_2_resourceBsequential_owshcilvwl_while_flzkvrshbq_readvariableop_2_resource_0"
>sequential_owshcilvwl_while_flzkvrshbq_readvariableop_resource@sequential_owshcilvwl_while_flzkvrshbq_readvariableop_resource_0"U
$sequential_owshcilvwl_while_identity-sequential/owshcilvwl/while/Identity:output:0"Y
&sequential_owshcilvwl_while_identity_1/sequential/owshcilvwl/while/Identity_1:output:0"Y
&sequential_owshcilvwl_while_identity_2/sequential/owshcilvwl/while/Identity_2:output:0"Y
&sequential_owshcilvwl_while_identity_3/sequential/owshcilvwl/while/Identity_3:output:0"Y
&sequential_owshcilvwl_while_identity_4/sequential/owshcilvwl/while/Identity_4:output:0"Y
&sequential_owshcilvwl_while_identity_5/sequential/owshcilvwl/while/Identity_5:output:0"
Asequential_owshcilvwl_while_sequential_owshcilvwl_strided_slice_1Csequential_owshcilvwl_while_sequential_owshcilvwl_strided_slice_1_0"
}sequential_owshcilvwl_while_tensorarrayv2read_tensorlistgetitem_sequential_owshcilvwl_tensorarrayunstack_tensorlistfromtensorsequential_owshcilvwl_while_tensorarrayv2read_tensorlistgetitem_sequential_owshcilvwl_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp=sequential/owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2|
<sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp<sequential/owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp2
>sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp>sequential/owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp2n
5sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp5sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp2r
7sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_17sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_12r
7sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_27sequential/owshcilvwl/while/flzkvrshbq/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
+__inference_sequential_layer_call_fn_655971

jfowsgvbzw
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
jfowsgvbzwunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
F__inference_sequential_layer_call_and_return_conditional_losses_6559362
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
jfowsgvbzw
Ò	
÷
F__inference_oaettnoaty_layer_call_and_return_conditional_losses_659245

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
owshcilvwl_while_body_6574792
.owshcilvwl_while_owshcilvwl_while_loop_counter8
4owshcilvwl_while_owshcilvwl_while_maximum_iterations 
owshcilvwl_while_placeholder"
owshcilvwl_while_placeholder_1"
owshcilvwl_while_placeholder_2"
owshcilvwl_while_placeholder_31
-owshcilvwl_while_owshcilvwl_strided_slice_1_0m
iowshcilvwl_while_tensorarrayv2read_tensorlistgetitem_owshcilvwl_tensorarrayunstack_tensorlistfromtensor_0O
<owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource_0:	 Q
>owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource_0:	 L
=owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource_0:	C
5owshcilvwl_while_flzkvrshbq_readvariableop_resource_0: E
7owshcilvwl_while_flzkvrshbq_readvariableop_1_resource_0: E
7owshcilvwl_while_flzkvrshbq_readvariableop_2_resource_0: 
owshcilvwl_while_identity
owshcilvwl_while_identity_1
owshcilvwl_while_identity_2
owshcilvwl_while_identity_3
owshcilvwl_while_identity_4
owshcilvwl_while_identity_5/
+owshcilvwl_while_owshcilvwl_strided_slice_1k
gowshcilvwl_while_tensorarrayv2read_tensorlistgetitem_owshcilvwl_tensorarrayunstack_tensorlistfromtensorM
:owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource:	 O
<owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource:	 J
;owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource:	A
3owshcilvwl_while_flzkvrshbq_readvariableop_resource: C
5owshcilvwl_while_flzkvrshbq_readvariableop_1_resource: C
5owshcilvwl_while_flzkvrshbq_readvariableop_2_resource: ¢2owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp¢1owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp¢3owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp¢*owshcilvwl/while/flzkvrshbq/ReadVariableOp¢,owshcilvwl/while/flzkvrshbq/ReadVariableOp_1¢,owshcilvwl/while/flzkvrshbq/ReadVariableOp_2Ù
Bowshcilvwl/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bowshcilvwl/while/TensorArrayV2Read/TensorListGetItem/element_shape
4owshcilvwl/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiowshcilvwl_while_tensorarrayv2read_tensorlistgetitem_owshcilvwl_tensorarrayunstack_tensorlistfromtensor_0owshcilvwl_while_placeholderKowshcilvwl/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4owshcilvwl/while/TensorArrayV2Read/TensorListGetItemä
1owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp<owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOpý
"owshcilvwl/while/flzkvrshbq/MatMulMatMul;owshcilvwl/while/TensorArrayV2Read/TensorListGetItem:item:09owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"owshcilvwl/while/flzkvrshbq/MatMulê
3owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp>owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOpæ
$owshcilvwl/while/flzkvrshbq/MatMul_1MatMulowshcilvwl_while_placeholder_2;owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$owshcilvwl/while/flzkvrshbq/MatMul_1Ü
owshcilvwl/while/flzkvrshbq/addAddV2,owshcilvwl/while/flzkvrshbq/MatMul:product:0.owshcilvwl/while/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
owshcilvwl/while/flzkvrshbq/addã
2owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp=owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOpé
#owshcilvwl/while/flzkvrshbq/BiasAddBiasAdd#owshcilvwl/while/flzkvrshbq/add:z:0:owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#owshcilvwl/while/flzkvrshbq/BiasAdd
+owshcilvwl/while/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+owshcilvwl/while/flzkvrshbq/split/split_dim¯
!owshcilvwl/while/flzkvrshbq/splitSplit4owshcilvwl/while/flzkvrshbq/split/split_dim:output:0,owshcilvwl/while/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!owshcilvwl/while/flzkvrshbq/splitÊ
*owshcilvwl/while/flzkvrshbq/ReadVariableOpReadVariableOp5owshcilvwl_while_flzkvrshbq_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*owshcilvwl/while/flzkvrshbq/ReadVariableOpÏ
owshcilvwl/while/flzkvrshbq/mulMul2owshcilvwl/while/flzkvrshbq/ReadVariableOp:value:0owshcilvwl_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
owshcilvwl/while/flzkvrshbq/mulÒ
!owshcilvwl/while/flzkvrshbq/add_1AddV2*owshcilvwl/while/flzkvrshbq/split:output:0#owshcilvwl/while/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/add_1®
#owshcilvwl/while/flzkvrshbq/SigmoidSigmoid%owshcilvwl/while/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#owshcilvwl/while/flzkvrshbq/SigmoidÐ
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_1ReadVariableOp7owshcilvwl_while_flzkvrshbq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_1Õ
!owshcilvwl/while/flzkvrshbq/mul_1Mul4owshcilvwl/while/flzkvrshbq/ReadVariableOp_1:value:0owshcilvwl_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/mul_1Ô
!owshcilvwl/while/flzkvrshbq/add_2AddV2*owshcilvwl/while/flzkvrshbq/split:output:1%owshcilvwl/while/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/add_2²
%owshcilvwl/while/flzkvrshbq/Sigmoid_1Sigmoid%owshcilvwl/while/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%owshcilvwl/while/flzkvrshbq/Sigmoid_1Ê
!owshcilvwl/while/flzkvrshbq/mul_2Mul)owshcilvwl/while/flzkvrshbq/Sigmoid_1:y:0owshcilvwl_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/mul_2ª
 owshcilvwl/while/flzkvrshbq/TanhTanh*owshcilvwl/while/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 owshcilvwl/while/flzkvrshbq/TanhÎ
!owshcilvwl/while/flzkvrshbq/mul_3Mul'owshcilvwl/while/flzkvrshbq/Sigmoid:y:0$owshcilvwl/while/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/mul_3Ï
!owshcilvwl/while/flzkvrshbq/add_3AddV2%owshcilvwl/while/flzkvrshbq/mul_2:z:0%owshcilvwl/while/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/add_3Ð
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_2ReadVariableOp7owshcilvwl_while_flzkvrshbq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_2Ü
!owshcilvwl/while/flzkvrshbq/mul_4Mul4owshcilvwl/while/flzkvrshbq/ReadVariableOp_2:value:0%owshcilvwl/while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/mul_4Ô
!owshcilvwl/while/flzkvrshbq/add_4AddV2*owshcilvwl/while/flzkvrshbq/split:output:3%owshcilvwl/while/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/add_4²
%owshcilvwl/while/flzkvrshbq/Sigmoid_2Sigmoid%owshcilvwl/while/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%owshcilvwl/while/flzkvrshbq/Sigmoid_2©
"owshcilvwl/while/flzkvrshbq/Tanh_1Tanh%owshcilvwl/while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"owshcilvwl/while/flzkvrshbq/Tanh_1Ò
!owshcilvwl/while/flzkvrshbq/mul_5Mul)owshcilvwl/while/flzkvrshbq/Sigmoid_2:y:0&owshcilvwl/while/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!owshcilvwl/while/flzkvrshbq/mul_5
5owshcilvwl/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemowshcilvwl_while_placeholder_1owshcilvwl_while_placeholder%owshcilvwl/while/flzkvrshbq/mul_5:z:0*
_output_shapes
: *
element_dtype027
5owshcilvwl/while/TensorArrayV2Write/TensorListSetItemr
owshcilvwl/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
owshcilvwl/while/add/y
owshcilvwl/while/addAddV2owshcilvwl_while_placeholderowshcilvwl/while/add/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/while/addv
owshcilvwl/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
owshcilvwl/while/add_1/y­
owshcilvwl/while/add_1AddV2.owshcilvwl_while_owshcilvwl_while_loop_counter!owshcilvwl/while/add_1/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/while/add_1©
owshcilvwl/while/IdentityIdentityowshcilvwl/while/add_1:z:03^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
owshcilvwl/while/IdentityÇ
owshcilvwl/while/Identity_1Identity4owshcilvwl_while_owshcilvwl_while_maximum_iterations3^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
owshcilvwl/while/Identity_1«
owshcilvwl/while/Identity_2Identityowshcilvwl/while/add:z:03^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
owshcilvwl/while/Identity_2Ø
owshcilvwl/while/Identity_3IdentityEowshcilvwl/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
owshcilvwl/while/Identity_3É
owshcilvwl/while/Identity_4Identity%owshcilvwl/while/flzkvrshbq/mul_5:z:03^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/while/Identity_4É
owshcilvwl/while/Identity_5Identity%owshcilvwl/while/flzkvrshbq/add_3:z:03^owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2^owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp4^owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp+^owshcilvwl/while/flzkvrshbq/ReadVariableOp-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_1-^owshcilvwl/while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/while/Identity_5"|
;owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource=owshcilvwl_while_flzkvrshbq_biasadd_readvariableop_resource_0"~
<owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource>owshcilvwl_while_flzkvrshbq_matmul_1_readvariableop_resource_0"z
:owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource<owshcilvwl_while_flzkvrshbq_matmul_readvariableop_resource_0"p
5owshcilvwl_while_flzkvrshbq_readvariableop_1_resource7owshcilvwl_while_flzkvrshbq_readvariableop_1_resource_0"p
5owshcilvwl_while_flzkvrshbq_readvariableop_2_resource7owshcilvwl_while_flzkvrshbq_readvariableop_2_resource_0"l
3owshcilvwl_while_flzkvrshbq_readvariableop_resource5owshcilvwl_while_flzkvrshbq_readvariableop_resource_0"?
owshcilvwl_while_identity"owshcilvwl/while/Identity:output:0"C
owshcilvwl_while_identity_1$owshcilvwl/while/Identity_1:output:0"C
owshcilvwl_while_identity_2$owshcilvwl/while/Identity_2:output:0"C
owshcilvwl_while_identity_3$owshcilvwl/while/Identity_3:output:0"C
owshcilvwl_while_identity_4$owshcilvwl/while/Identity_4:output:0"C
owshcilvwl_while_identity_5$owshcilvwl/while/Identity_5:output:0"\
+owshcilvwl_while_owshcilvwl_strided_slice_1-owshcilvwl_while_owshcilvwl_strided_slice_1_0"Ô
gowshcilvwl_while_tensorarrayv2read_tensorlistgetitem_owshcilvwl_tensorarrayunstack_tensorlistfromtensoriowshcilvwl_while_tensorarrayv2read_tensorlistgetitem_owshcilvwl_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2owshcilvwl/while/flzkvrshbq/BiasAdd/ReadVariableOp2f
1owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp1owshcilvwl/while/flzkvrshbq/MatMul/ReadVariableOp2j
3owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp3owshcilvwl/while/flzkvrshbq/MatMul_1/ReadVariableOp2X
*owshcilvwl/while/flzkvrshbq/ReadVariableOp*owshcilvwl/while/flzkvrshbq/ReadVariableOp2\
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_1,owshcilvwl/while/flzkvrshbq/ReadVariableOp_12\
,owshcilvwl/while/flzkvrshbq/ReadVariableOp_2,owshcilvwl/while/flzkvrshbq/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_654817
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_flzkvrshbq_654841_0:	 ,
while_flzkvrshbq_654843_0:	 (
while_flzkvrshbq_654845_0:	'
while_flzkvrshbq_654847_0: '
while_flzkvrshbq_654849_0: '
while_flzkvrshbq_654851_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_flzkvrshbq_654841:	 *
while_flzkvrshbq_654843:	 &
while_flzkvrshbq_654845:	%
while_flzkvrshbq_654847: %
while_flzkvrshbq_654849: %
while_flzkvrshbq_654851: ¢(while/flzkvrshbq/StatefulPartitionedCallÃ
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
(while/flzkvrshbq/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_flzkvrshbq_654841_0while_flzkvrshbq_654843_0while_flzkvrshbq_654845_0while_flzkvrshbq_654847_0while_flzkvrshbq_654849_0while_flzkvrshbq_654851_0*
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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_6547972*
(while/flzkvrshbq/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/flzkvrshbq/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/flzkvrshbq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/flzkvrshbq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/flzkvrshbq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/flzkvrshbq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/flzkvrshbq/StatefulPartitionedCall:output:1)^while/flzkvrshbq/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/flzkvrshbq/StatefulPartitionedCall:output:2)^while/flzkvrshbq/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_flzkvrshbq_654841while_flzkvrshbq_654841_0"4
while_flzkvrshbq_654843while_flzkvrshbq_654843_0"4
while_flzkvrshbq_654845while_flzkvrshbq_654845_0"4
while_flzkvrshbq_654847while_flzkvrshbq_654847_0"4
while_flzkvrshbq_654849while_flzkvrshbq_654849_0"4
while_flzkvrshbq_654851while_flzkvrshbq_654851_0")
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
(while/flzkvrshbq/StatefulPartitionedCall(while/flzkvrshbq/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_657796
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_657796___redundant_placeholder04
0while_while_cond_657796___redundant_placeholder14
0while_while_cond_657796___redundant_placeholder24
0while_while_cond_657796___redundant_placeholder34
0while_while_cond_657796___redundant_placeholder44
0while_while_cond_657796___redundant_placeholder54
0while_while_cond_657796___redundant_placeholder6
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
while_body_658945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_flzkvrshbq_matmul_readvariableop_resource_0:	 F
3while_flzkvrshbq_matmul_1_readvariableop_resource_0:	 A
2while_flzkvrshbq_biasadd_readvariableop_resource_0:	8
*while_flzkvrshbq_readvariableop_resource_0: :
,while_flzkvrshbq_readvariableop_1_resource_0: :
,while_flzkvrshbq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_flzkvrshbq_matmul_readvariableop_resource:	 D
1while_flzkvrshbq_matmul_1_readvariableop_resource:	 ?
0while_flzkvrshbq_biasadd_readvariableop_resource:	6
(while_flzkvrshbq_readvariableop_resource: 8
*while_flzkvrshbq_readvariableop_1_resource: 8
*while_flzkvrshbq_readvariableop_2_resource: ¢'while/flzkvrshbq/BiasAdd/ReadVariableOp¢&while/flzkvrshbq/MatMul/ReadVariableOp¢(while/flzkvrshbq/MatMul_1/ReadVariableOp¢while/flzkvrshbq/ReadVariableOp¢!while/flzkvrshbq/ReadVariableOp_1¢!while/flzkvrshbq/ReadVariableOp_2Ã
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
&while/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp1while_flzkvrshbq_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/flzkvrshbq/MatMul/ReadVariableOpÑ
while/flzkvrshbq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMulÉ
(while/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp3while_flzkvrshbq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/flzkvrshbq/MatMul_1/ReadVariableOpº
while/flzkvrshbq/MatMul_1MatMulwhile_placeholder_20while/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMul_1°
while/flzkvrshbq/addAddV2!while/flzkvrshbq/MatMul:product:0#while/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/addÂ
'while/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp2while_flzkvrshbq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/flzkvrshbq/BiasAdd/ReadVariableOp½
while/flzkvrshbq/BiasAddBiasAddwhile/flzkvrshbq/add:z:0/while/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/BiasAdd
 while/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/flzkvrshbq/split/split_dim
while/flzkvrshbq/splitSplit)while/flzkvrshbq/split/split_dim:output:0!while/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/flzkvrshbq/split©
while/flzkvrshbq/ReadVariableOpReadVariableOp*while_flzkvrshbq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/flzkvrshbq/ReadVariableOp£
while/flzkvrshbq/mulMul'while/flzkvrshbq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul¦
while/flzkvrshbq/add_1AddV2while/flzkvrshbq/split:output:0while/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_1
while/flzkvrshbq/SigmoidSigmoidwhile/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid¯
!while/flzkvrshbq/ReadVariableOp_1ReadVariableOp,while_flzkvrshbq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_1©
while/flzkvrshbq/mul_1Mul)while/flzkvrshbq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_1¨
while/flzkvrshbq/add_2AddV2while/flzkvrshbq/split:output:1while/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_2
while/flzkvrshbq/Sigmoid_1Sigmoidwhile/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_1
while/flzkvrshbq/mul_2Mulwhile/flzkvrshbq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_2
while/flzkvrshbq/TanhTanhwhile/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh¢
while/flzkvrshbq/mul_3Mulwhile/flzkvrshbq/Sigmoid:y:0while/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_3£
while/flzkvrshbq/add_3AddV2while/flzkvrshbq/mul_2:z:0while/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_3¯
!while/flzkvrshbq/ReadVariableOp_2ReadVariableOp,while_flzkvrshbq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_2°
while/flzkvrshbq/mul_4Mul)while/flzkvrshbq/ReadVariableOp_2:value:0while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_4¨
while/flzkvrshbq/add_4AddV2while/flzkvrshbq/split:output:3while/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_4
while/flzkvrshbq/Sigmoid_2Sigmoidwhile/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_2
while/flzkvrshbq/Tanh_1Tanhwhile/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh_1¦
while/flzkvrshbq/mul_5Mulwhile/flzkvrshbq/Sigmoid_2:y:0while/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/flzkvrshbq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/flzkvrshbq/mul_5:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/flzkvrshbq/add_3:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_flzkvrshbq_biasadd_readvariableop_resource2while_flzkvrshbq_biasadd_readvariableop_resource_0"h
1while_flzkvrshbq_matmul_1_readvariableop_resource3while_flzkvrshbq_matmul_1_readvariableop_resource_0"d
/while_flzkvrshbq_matmul_readvariableop_resource1while_flzkvrshbq_matmul_readvariableop_resource_0"Z
*while_flzkvrshbq_readvariableop_1_resource,while_flzkvrshbq_readvariableop_1_resource_0"Z
*while_flzkvrshbq_readvariableop_2_resource,while_flzkvrshbq_readvariableop_2_resource_0"V
(while_flzkvrshbq_readvariableop_resource*while_flzkvrshbq_readvariableop_resource_0")
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
'while/flzkvrshbq/BiasAdd/ReadVariableOp'while/flzkvrshbq/BiasAdd/ReadVariableOp2P
&while/flzkvrshbq/MatMul/ReadVariableOp&while/flzkvrshbq/MatMul/ReadVariableOp2T
(while/flzkvrshbq/MatMul_1/ReadVariableOp(while/flzkvrshbq/MatMul_1/ReadVariableOp2B
while/flzkvrshbq/ReadVariableOpwhile/flzkvrshbq/ReadVariableOp2F
!while/flzkvrshbq/ReadVariableOp_1!while/flzkvrshbq/ReadVariableOp_12F
!while/flzkvrshbq/ReadVariableOp_2!while/flzkvrshbq/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_658764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_658764___redundant_placeholder04
0while_while_cond_658764___redundant_placeholder14
0while_while_cond_658764___redundant_placeholder24
0while_while_cond_658764___redundant_placeholder34
0while_while_cond_658764___redundant_placeholder44
0while_while_cond_658764___redundant_placeholder54
0while_while_cond_658764___redundant_placeholder6
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
'sequential_owshcilvwl_while_cond_653844H
Dsequential_owshcilvwl_while_sequential_owshcilvwl_while_loop_counterN
Jsequential_owshcilvwl_while_sequential_owshcilvwl_while_maximum_iterations+
'sequential_owshcilvwl_while_placeholder-
)sequential_owshcilvwl_while_placeholder_1-
)sequential_owshcilvwl_while_placeholder_2-
)sequential_owshcilvwl_while_placeholder_3J
Fsequential_owshcilvwl_while_less_sequential_owshcilvwl_strided_slice_1`
\sequential_owshcilvwl_while_sequential_owshcilvwl_while_cond_653844___redundant_placeholder0`
\sequential_owshcilvwl_while_sequential_owshcilvwl_while_cond_653844___redundant_placeholder1`
\sequential_owshcilvwl_while_sequential_owshcilvwl_while_cond_653844___redundant_placeholder2`
\sequential_owshcilvwl_while_sequential_owshcilvwl_while_cond_653844___redundant_placeholder3`
\sequential_owshcilvwl_while_sequential_owshcilvwl_while_cond_653844___redundant_placeholder4`
\sequential_owshcilvwl_while_sequential_owshcilvwl_while_cond_653844___redundant_placeholder5`
\sequential_owshcilvwl_while_sequential_owshcilvwl_while_cond_653844___redundant_placeholder6(
$sequential_owshcilvwl_while_identity
Þ
 sequential/owshcilvwl/while/LessLess'sequential_owshcilvwl_while_placeholderFsequential_owshcilvwl_while_less_sequential_owshcilvwl_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/owshcilvwl/while/Less
$sequential/owshcilvwl/while/IdentityIdentity$sequential/owshcilvwl/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/owshcilvwl/while/Identity"U
$sequential_owshcilvwl_while_identity-sequential/owshcilvwl/while/Identity:output:0*(
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
ÙÊ

F__inference_sequential_layer_call_and_return_conditional_losses_657182

inputsL
6bpstkcuudk_conv1d_expanddims_1_readvariableop_resource:K
=bpstkcuudk_squeeze_batch_dims_biasadd_readvariableop_resource:G
4osutmzfngz_dsycfvoega_matmul_readvariableop_resource:	I
6osutmzfngz_dsycfvoega_matmul_1_readvariableop_resource:	 D
5osutmzfngz_dsycfvoega_biasadd_readvariableop_resource:	;
-osutmzfngz_dsycfvoega_readvariableop_resource: =
/osutmzfngz_dsycfvoega_readvariableop_1_resource: =
/osutmzfngz_dsycfvoega_readvariableop_2_resource: G
4owshcilvwl_flzkvrshbq_matmul_readvariableop_resource:	 I
6owshcilvwl_flzkvrshbq_matmul_1_readvariableop_resource:	 D
5owshcilvwl_flzkvrshbq_biasadd_readvariableop_resource:	;
-owshcilvwl_flzkvrshbq_readvariableop_resource: =
/owshcilvwl_flzkvrshbq_readvariableop_1_resource: =
/owshcilvwl_flzkvrshbq_readvariableop_2_resource: ;
)oaettnoaty_matmul_readvariableop_resource: 8
*oaettnoaty_biasadd_readvariableop_resource:
identity¢-bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp¢4bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp¢!oaettnoaty/BiasAdd/ReadVariableOp¢ oaettnoaty/MatMul/ReadVariableOp¢,osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp¢+osutmzfngz/dsycfvoega/MatMul/ReadVariableOp¢-osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp¢$osutmzfngz/dsycfvoega/ReadVariableOp¢&osutmzfngz/dsycfvoega/ReadVariableOp_1¢&osutmzfngz/dsycfvoega/ReadVariableOp_2¢osutmzfngz/while¢,owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp¢+owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp¢-owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp¢$owshcilvwl/flzkvrshbq/ReadVariableOp¢&owshcilvwl/flzkvrshbq/ReadVariableOp_1¢&owshcilvwl/flzkvrshbq/ReadVariableOp_2¢owshcilvwl/while
 bpstkcuudk/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 bpstkcuudk/conv1d/ExpandDims/dim»
bpstkcuudk/conv1d/ExpandDims
ExpandDimsinputs)bpstkcuudk/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
bpstkcuudk/conv1d/ExpandDimsÙ
-bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6bpstkcuudk_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp
"bpstkcuudk/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"bpstkcuudk/conv1d/ExpandDims_1/dimã
bpstkcuudk/conv1d/ExpandDims_1
ExpandDims5bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp:value:0+bpstkcuudk/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
bpstkcuudk/conv1d/ExpandDims_1
bpstkcuudk/conv1d/ShapeShape%bpstkcuudk/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
bpstkcuudk/conv1d/Shape
%bpstkcuudk/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%bpstkcuudk/conv1d/strided_slice/stack¥
'bpstkcuudk/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'bpstkcuudk/conv1d/strided_slice/stack_1
'bpstkcuudk/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'bpstkcuudk/conv1d/strided_slice/stack_2Ì
bpstkcuudk/conv1d/strided_sliceStridedSlice bpstkcuudk/conv1d/Shape:output:0.bpstkcuudk/conv1d/strided_slice/stack:output:00bpstkcuudk/conv1d/strided_slice/stack_1:output:00bpstkcuudk/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
bpstkcuudk/conv1d/strided_slice
bpstkcuudk/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
bpstkcuudk/conv1d/Reshape/shapeÌ
bpstkcuudk/conv1d/ReshapeReshape%bpstkcuudk/conv1d/ExpandDims:output:0(bpstkcuudk/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bpstkcuudk/conv1d/Reshapeî
bpstkcuudk/conv1d/Conv2DConv2D"bpstkcuudk/conv1d/Reshape:output:0'bpstkcuudk/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
bpstkcuudk/conv1d/Conv2D
!bpstkcuudk/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!bpstkcuudk/conv1d/concat/values_1
bpstkcuudk/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
bpstkcuudk/conv1d/concat/axisì
bpstkcuudk/conv1d/concatConcatV2(bpstkcuudk/conv1d/strided_slice:output:0*bpstkcuudk/conv1d/concat/values_1:output:0&bpstkcuudk/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
bpstkcuudk/conv1d/concatÉ
bpstkcuudk/conv1d/Reshape_1Reshape!bpstkcuudk/conv1d/Conv2D:output:0!bpstkcuudk/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
bpstkcuudk/conv1d/Reshape_1Á
bpstkcuudk/conv1d/SqueezeSqueeze$bpstkcuudk/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
bpstkcuudk/conv1d/Squeeze
#bpstkcuudk/squeeze_batch_dims/ShapeShape"bpstkcuudk/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#bpstkcuudk/squeeze_batch_dims/Shape°
1bpstkcuudk/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1bpstkcuudk/squeeze_batch_dims/strided_slice/stack½
3bpstkcuudk/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3bpstkcuudk/squeeze_batch_dims/strided_slice/stack_1´
3bpstkcuudk/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3bpstkcuudk/squeeze_batch_dims/strided_slice/stack_2
+bpstkcuudk/squeeze_batch_dims/strided_sliceStridedSlice,bpstkcuudk/squeeze_batch_dims/Shape:output:0:bpstkcuudk/squeeze_batch_dims/strided_slice/stack:output:0<bpstkcuudk/squeeze_batch_dims/strided_slice/stack_1:output:0<bpstkcuudk/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+bpstkcuudk/squeeze_batch_dims/strided_slice¯
+bpstkcuudk/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+bpstkcuudk/squeeze_batch_dims/Reshape/shapeé
%bpstkcuudk/squeeze_batch_dims/ReshapeReshape"bpstkcuudk/conv1d/Squeeze:output:04bpstkcuudk/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%bpstkcuudk/squeeze_batch_dims/Reshapeæ
4bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=bpstkcuudk_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%bpstkcuudk/squeeze_batch_dims/BiasAddBiasAdd.bpstkcuudk/squeeze_batch_dims/Reshape:output:0<bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%bpstkcuudk/squeeze_batch_dims/BiasAdd¯
-bpstkcuudk/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-bpstkcuudk/squeeze_batch_dims/concat/values_1¡
)bpstkcuudk/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)bpstkcuudk/squeeze_batch_dims/concat/axis¨
$bpstkcuudk/squeeze_batch_dims/concatConcatV24bpstkcuudk/squeeze_batch_dims/strided_slice:output:06bpstkcuudk/squeeze_batch_dims/concat/values_1:output:02bpstkcuudk/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$bpstkcuudk/squeeze_batch_dims/concatö
'bpstkcuudk/squeeze_batch_dims/Reshape_1Reshape.bpstkcuudk/squeeze_batch_dims/BiasAdd:output:0-bpstkcuudk/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'bpstkcuudk/squeeze_batch_dims/Reshape_1
xlcvyoxoxq/ShapeShape0bpstkcuudk/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
xlcvyoxoxq/Shape
xlcvyoxoxq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
xlcvyoxoxq/strided_slice/stack
 xlcvyoxoxq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 xlcvyoxoxq/strided_slice/stack_1
 xlcvyoxoxq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 xlcvyoxoxq/strided_slice/stack_2¤
xlcvyoxoxq/strided_sliceStridedSlicexlcvyoxoxq/Shape:output:0'xlcvyoxoxq/strided_slice/stack:output:0)xlcvyoxoxq/strided_slice/stack_1:output:0)xlcvyoxoxq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
xlcvyoxoxq/strided_slicez
xlcvyoxoxq/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
xlcvyoxoxq/Reshape/shape/1z
xlcvyoxoxq/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
xlcvyoxoxq/Reshape/shape/2×
xlcvyoxoxq/Reshape/shapePack!xlcvyoxoxq/strided_slice:output:0#xlcvyoxoxq/Reshape/shape/1:output:0#xlcvyoxoxq/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
xlcvyoxoxq/Reshape/shape¾
xlcvyoxoxq/ReshapeReshape0bpstkcuudk/squeeze_batch_dims/Reshape_1:output:0!xlcvyoxoxq/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xlcvyoxoxq/Reshapeo
osutmzfngz/ShapeShapexlcvyoxoxq/Reshape:output:0*
T0*
_output_shapes
:2
osutmzfngz/Shape
osutmzfngz/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
osutmzfngz/strided_slice/stack
 osutmzfngz/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 osutmzfngz/strided_slice/stack_1
 osutmzfngz/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 osutmzfngz/strided_slice/stack_2¤
osutmzfngz/strided_sliceStridedSliceosutmzfngz/Shape:output:0'osutmzfngz/strided_slice/stack:output:0)osutmzfngz/strided_slice/stack_1:output:0)osutmzfngz/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
osutmzfngz/strided_slicer
osutmzfngz/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/zeros/mul/y
osutmzfngz/zeros/mulMul!osutmzfngz/strided_slice:output:0osutmzfngz/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/zeros/mulu
osutmzfngz/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
osutmzfngz/zeros/Less/y
osutmzfngz/zeros/LessLessosutmzfngz/zeros/mul:z:0 osutmzfngz/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/zeros/Lessx
osutmzfngz/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/zeros/packed/1¯
osutmzfngz/zeros/packedPack!osutmzfngz/strided_slice:output:0"osutmzfngz/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
osutmzfngz/zeros/packedu
osutmzfngz/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
osutmzfngz/zeros/Const¡
osutmzfngz/zerosFill osutmzfngz/zeros/packed:output:0osutmzfngz/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/zerosv
osutmzfngz/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/zeros_1/mul/y
osutmzfngz/zeros_1/mulMul!osutmzfngz/strided_slice:output:0!osutmzfngz/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/zeros_1/muly
osutmzfngz/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
osutmzfngz/zeros_1/Less/y
osutmzfngz/zeros_1/LessLessosutmzfngz/zeros_1/mul:z:0"osutmzfngz/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/zeros_1/Less|
osutmzfngz/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/zeros_1/packed/1µ
osutmzfngz/zeros_1/packedPack!osutmzfngz/strided_slice:output:0$osutmzfngz/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
osutmzfngz/zeros_1/packedy
osutmzfngz/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
osutmzfngz/zeros_1/Const©
osutmzfngz/zeros_1Fill"osutmzfngz/zeros_1/packed:output:0!osutmzfngz/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/zeros_1
osutmzfngz/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
osutmzfngz/transpose/perm°
osutmzfngz/transpose	Transposexlcvyoxoxq/Reshape:output:0"osutmzfngz/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
osutmzfngz/transposep
osutmzfngz/Shape_1Shapeosutmzfngz/transpose:y:0*
T0*
_output_shapes
:2
osutmzfngz/Shape_1
 osutmzfngz/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 osutmzfngz/strided_slice_1/stack
"osutmzfngz/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"osutmzfngz/strided_slice_1/stack_1
"osutmzfngz/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"osutmzfngz/strided_slice_1/stack_2°
osutmzfngz/strided_slice_1StridedSliceosutmzfngz/Shape_1:output:0)osutmzfngz/strided_slice_1/stack:output:0+osutmzfngz/strided_slice_1/stack_1:output:0+osutmzfngz/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
osutmzfngz/strided_slice_1
&osutmzfngz/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&osutmzfngz/TensorArrayV2/element_shapeÞ
osutmzfngz/TensorArrayV2TensorListReserve/osutmzfngz/TensorArrayV2/element_shape:output:0#osutmzfngz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
osutmzfngz/TensorArrayV2Õ
@osutmzfngz/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@osutmzfngz/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2osutmzfngz/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorosutmzfngz/transpose:y:0Iosutmzfngz/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2osutmzfngz/TensorArrayUnstack/TensorListFromTensor
 osutmzfngz/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 osutmzfngz/strided_slice_2/stack
"osutmzfngz/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"osutmzfngz/strided_slice_2/stack_1
"osutmzfngz/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"osutmzfngz/strided_slice_2/stack_2¾
osutmzfngz/strided_slice_2StridedSliceosutmzfngz/transpose:y:0)osutmzfngz/strided_slice_2/stack:output:0+osutmzfngz/strided_slice_2/stack_1:output:0+osutmzfngz/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
osutmzfngz/strided_slice_2Ð
+osutmzfngz/dsycfvoega/MatMul/ReadVariableOpReadVariableOp4osutmzfngz_dsycfvoega_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+osutmzfngz/dsycfvoega/MatMul/ReadVariableOpÓ
osutmzfngz/dsycfvoega/MatMulMatMul#osutmzfngz/strided_slice_2:output:03osutmzfngz/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
osutmzfngz/dsycfvoega/MatMulÖ
-osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp6osutmzfngz_dsycfvoega_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOpÏ
osutmzfngz/dsycfvoega/MatMul_1MatMulosutmzfngz/zeros:output:05osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
osutmzfngz/dsycfvoega/MatMul_1Ä
osutmzfngz/dsycfvoega/addAddV2&osutmzfngz/dsycfvoega/MatMul:product:0(osutmzfngz/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
osutmzfngz/dsycfvoega/addÏ
,osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp5osutmzfngz_dsycfvoega_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOpÑ
osutmzfngz/dsycfvoega/BiasAddBiasAddosutmzfngz/dsycfvoega/add:z:04osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
osutmzfngz/dsycfvoega/BiasAdd
%osutmzfngz/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%osutmzfngz/dsycfvoega/split/split_dim
osutmzfngz/dsycfvoega/splitSplit.osutmzfngz/dsycfvoega/split/split_dim:output:0&osutmzfngz/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
osutmzfngz/dsycfvoega/split¶
$osutmzfngz/dsycfvoega/ReadVariableOpReadVariableOp-osutmzfngz_dsycfvoega_readvariableop_resource*
_output_shapes
: *
dtype02&
$osutmzfngz/dsycfvoega/ReadVariableOpº
osutmzfngz/dsycfvoega/mulMul,osutmzfngz/dsycfvoega/ReadVariableOp:value:0osutmzfngz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mulº
osutmzfngz/dsycfvoega/add_1AddV2$osutmzfngz/dsycfvoega/split:output:0osutmzfngz/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/add_1
osutmzfngz/dsycfvoega/SigmoidSigmoidosutmzfngz/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/Sigmoid¼
&osutmzfngz/dsycfvoega/ReadVariableOp_1ReadVariableOp/osutmzfngz_dsycfvoega_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&osutmzfngz/dsycfvoega/ReadVariableOp_1À
osutmzfngz/dsycfvoega/mul_1Mul.osutmzfngz/dsycfvoega/ReadVariableOp_1:value:0osutmzfngz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mul_1¼
osutmzfngz/dsycfvoega/add_2AddV2$osutmzfngz/dsycfvoega/split:output:1osutmzfngz/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/add_2 
osutmzfngz/dsycfvoega/Sigmoid_1Sigmoidosutmzfngz/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
osutmzfngz/dsycfvoega/Sigmoid_1µ
osutmzfngz/dsycfvoega/mul_2Mul#osutmzfngz/dsycfvoega/Sigmoid_1:y:0osutmzfngz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mul_2
osutmzfngz/dsycfvoega/TanhTanh$osutmzfngz/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/Tanh¶
osutmzfngz/dsycfvoega/mul_3Mul!osutmzfngz/dsycfvoega/Sigmoid:y:0osutmzfngz/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mul_3·
osutmzfngz/dsycfvoega/add_3AddV2osutmzfngz/dsycfvoega/mul_2:z:0osutmzfngz/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/add_3¼
&osutmzfngz/dsycfvoega/ReadVariableOp_2ReadVariableOp/osutmzfngz_dsycfvoega_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&osutmzfngz/dsycfvoega/ReadVariableOp_2Ä
osutmzfngz/dsycfvoega/mul_4Mul.osutmzfngz/dsycfvoega/ReadVariableOp_2:value:0osutmzfngz/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mul_4¼
osutmzfngz/dsycfvoega/add_4AddV2$osutmzfngz/dsycfvoega/split:output:3osutmzfngz/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/add_4 
osutmzfngz/dsycfvoega/Sigmoid_2Sigmoidosutmzfngz/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
osutmzfngz/dsycfvoega/Sigmoid_2
osutmzfngz/dsycfvoega/Tanh_1Tanhosutmzfngz/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/Tanh_1º
osutmzfngz/dsycfvoega/mul_5Mul#osutmzfngz/dsycfvoega/Sigmoid_2:y:0 osutmzfngz/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mul_5¥
(osutmzfngz/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(osutmzfngz/TensorArrayV2_1/element_shapeä
osutmzfngz/TensorArrayV2_1TensorListReserve1osutmzfngz/TensorArrayV2_1/element_shape:output:0#osutmzfngz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
osutmzfngz/TensorArrayV2_1d
osutmzfngz/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/time
#osutmzfngz/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#osutmzfngz/while/maximum_iterations
osutmzfngz/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/while/loop_counter°
osutmzfngz/whileWhile&osutmzfngz/while/loop_counter:output:0,osutmzfngz/while/maximum_iterations:output:0osutmzfngz/time:output:0#osutmzfngz/TensorArrayV2_1:handle:0osutmzfngz/zeros:output:0osutmzfngz/zeros_1:output:0#osutmzfngz/strided_slice_1:output:0Bosutmzfngz/TensorArrayUnstack/TensorListFromTensor:output_handle:04osutmzfngz_dsycfvoega_matmul_readvariableop_resource6osutmzfngz_dsycfvoega_matmul_1_readvariableop_resource5osutmzfngz_dsycfvoega_biasadd_readvariableop_resource-osutmzfngz_dsycfvoega_readvariableop_resource/osutmzfngz_dsycfvoega_readvariableop_1_resource/osutmzfngz_dsycfvoega_readvariableop_2_resource*
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
osutmzfngz_while_body_656899*(
cond R
osutmzfngz_while_cond_656898*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
osutmzfngz/whileË
;osutmzfngz/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;osutmzfngz/TensorArrayV2Stack/TensorListStack/element_shape
-osutmzfngz/TensorArrayV2Stack/TensorListStackTensorListStackosutmzfngz/while:output:3Dosutmzfngz/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-osutmzfngz/TensorArrayV2Stack/TensorListStack
 osutmzfngz/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 osutmzfngz/strided_slice_3/stack
"osutmzfngz/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"osutmzfngz/strided_slice_3/stack_1
"osutmzfngz/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"osutmzfngz/strided_slice_3/stack_2Ü
osutmzfngz/strided_slice_3StridedSlice6osutmzfngz/TensorArrayV2Stack/TensorListStack:tensor:0)osutmzfngz/strided_slice_3/stack:output:0+osutmzfngz/strided_slice_3/stack_1:output:0+osutmzfngz/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
osutmzfngz/strided_slice_3
osutmzfngz/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
osutmzfngz/transpose_1/permÑ
osutmzfngz/transpose_1	Transpose6osutmzfngz/TensorArrayV2Stack/TensorListStack:tensor:0$osutmzfngz/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/transpose_1n
owshcilvwl/ShapeShapeosutmzfngz/transpose_1:y:0*
T0*
_output_shapes
:2
owshcilvwl/Shape
owshcilvwl/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
owshcilvwl/strided_slice/stack
 owshcilvwl/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 owshcilvwl/strided_slice/stack_1
 owshcilvwl/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 owshcilvwl/strided_slice/stack_2¤
owshcilvwl/strided_sliceStridedSliceowshcilvwl/Shape:output:0'owshcilvwl/strided_slice/stack:output:0)owshcilvwl/strided_slice/stack_1:output:0)owshcilvwl/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
owshcilvwl/strided_slicer
owshcilvwl/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/zeros/mul/y
owshcilvwl/zeros/mulMul!owshcilvwl/strided_slice:output:0owshcilvwl/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/zeros/mulu
owshcilvwl/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
owshcilvwl/zeros/Less/y
owshcilvwl/zeros/LessLessowshcilvwl/zeros/mul:z:0 owshcilvwl/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/zeros/Lessx
owshcilvwl/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/zeros/packed/1¯
owshcilvwl/zeros/packedPack!owshcilvwl/strided_slice:output:0"owshcilvwl/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
owshcilvwl/zeros/packedu
owshcilvwl/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
owshcilvwl/zeros/Const¡
owshcilvwl/zerosFill owshcilvwl/zeros/packed:output:0owshcilvwl/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/zerosv
owshcilvwl/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/zeros_1/mul/y
owshcilvwl/zeros_1/mulMul!owshcilvwl/strided_slice:output:0!owshcilvwl/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/zeros_1/muly
owshcilvwl/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
owshcilvwl/zeros_1/Less/y
owshcilvwl/zeros_1/LessLessowshcilvwl/zeros_1/mul:z:0"owshcilvwl/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/zeros_1/Less|
owshcilvwl/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/zeros_1/packed/1µ
owshcilvwl/zeros_1/packedPack!owshcilvwl/strided_slice:output:0$owshcilvwl/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
owshcilvwl/zeros_1/packedy
owshcilvwl/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
owshcilvwl/zeros_1/Const©
owshcilvwl/zeros_1Fill"owshcilvwl/zeros_1/packed:output:0!owshcilvwl/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/zeros_1
owshcilvwl/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
owshcilvwl/transpose/perm¯
owshcilvwl/transpose	Transposeosutmzfngz/transpose_1:y:0"owshcilvwl/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/transposep
owshcilvwl/Shape_1Shapeowshcilvwl/transpose:y:0*
T0*
_output_shapes
:2
owshcilvwl/Shape_1
 owshcilvwl/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 owshcilvwl/strided_slice_1/stack
"owshcilvwl/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"owshcilvwl/strided_slice_1/stack_1
"owshcilvwl/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"owshcilvwl/strided_slice_1/stack_2°
owshcilvwl/strided_slice_1StridedSliceowshcilvwl/Shape_1:output:0)owshcilvwl/strided_slice_1/stack:output:0+owshcilvwl/strided_slice_1/stack_1:output:0+owshcilvwl/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
owshcilvwl/strided_slice_1
&owshcilvwl/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&owshcilvwl/TensorArrayV2/element_shapeÞ
owshcilvwl/TensorArrayV2TensorListReserve/owshcilvwl/TensorArrayV2/element_shape:output:0#owshcilvwl/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
owshcilvwl/TensorArrayV2Õ
@owshcilvwl/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@owshcilvwl/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2owshcilvwl/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorowshcilvwl/transpose:y:0Iowshcilvwl/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2owshcilvwl/TensorArrayUnstack/TensorListFromTensor
 owshcilvwl/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 owshcilvwl/strided_slice_2/stack
"owshcilvwl/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"owshcilvwl/strided_slice_2/stack_1
"owshcilvwl/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"owshcilvwl/strided_slice_2/stack_2¾
owshcilvwl/strided_slice_2StridedSliceowshcilvwl/transpose:y:0)owshcilvwl/strided_slice_2/stack:output:0+owshcilvwl/strided_slice_2/stack_1:output:0+owshcilvwl/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
owshcilvwl/strided_slice_2Ð
+owshcilvwl/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp4owshcilvwl_flzkvrshbq_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+owshcilvwl/flzkvrshbq/MatMul/ReadVariableOpÓ
owshcilvwl/flzkvrshbq/MatMulMatMul#owshcilvwl/strided_slice_2:output:03owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
owshcilvwl/flzkvrshbq/MatMulÖ
-owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp6owshcilvwl_flzkvrshbq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOpÏ
owshcilvwl/flzkvrshbq/MatMul_1MatMulowshcilvwl/zeros:output:05owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
owshcilvwl/flzkvrshbq/MatMul_1Ä
owshcilvwl/flzkvrshbq/addAddV2&owshcilvwl/flzkvrshbq/MatMul:product:0(owshcilvwl/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
owshcilvwl/flzkvrshbq/addÏ
,owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp5owshcilvwl_flzkvrshbq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOpÑ
owshcilvwl/flzkvrshbq/BiasAddBiasAddowshcilvwl/flzkvrshbq/add:z:04owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
owshcilvwl/flzkvrshbq/BiasAdd
%owshcilvwl/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%owshcilvwl/flzkvrshbq/split/split_dim
owshcilvwl/flzkvrshbq/splitSplit.owshcilvwl/flzkvrshbq/split/split_dim:output:0&owshcilvwl/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
owshcilvwl/flzkvrshbq/split¶
$owshcilvwl/flzkvrshbq/ReadVariableOpReadVariableOp-owshcilvwl_flzkvrshbq_readvariableop_resource*
_output_shapes
: *
dtype02&
$owshcilvwl/flzkvrshbq/ReadVariableOpº
owshcilvwl/flzkvrshbq/mulMul,owshcilvwl/flzkvrshbq/ReadVariableOp:value:0owshcilvwl/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mulº
owshcilvwl/flzkvrshbq/add_1AddV2$owshcilvwl/flzkvrshbq/split:output:0owshcilvwl/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/add_1
owshcilvwl/flzkvrshbq/SigmoidSigmoidowshcilvwl/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/Sigmoid¼
&owshcilvwl/flzkvrshbq/ReadVariableOp_1ReadVariableOp/owshcilvwl_flzkvrshbq_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&owshcilvwl/flzkvrshbq/ReadVariableOp_1À
owshcilvwl/flzkvrshbq/mul_1Mul.owshcilvwl/flzkvrshbq/ReadVariableOp_1:value:0owshcilvwl/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mul_1¼
owshcilvwl/flzkvrshbq/add_2AddV2$owshcilvwl/flzkvrshbq/split:output:1owshcilvwl/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/add_2 
owshcilvwl/flzkvrshbq/Sigmoid_1Sigmoidowshcilvwl/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
owshcilvwl/flzkvrshbq/Sigmoid_1µ
owshcilvwl/flzkvrshbq/mul_2Mul#owshcilvwl/flzkvrshbq/Sigmoid_1:y:0owshcilvwl/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mul_2
owshcilvwl/flzkvrshbq/TanhTanh$owshcilvwl/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/Tanh¶
owshcilvwl/flzkvrshbq/mul_3Mul!owshcilvwl/flzkvrshbq/Sigmoid:y:0owshcilvwl/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mul_3·
owshcilvwl/flzkvrshbq/add_3AddV2owshcilvwl/flzkvrshbq/mul_2:z:0owshcilvwl/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/add_3¼
&owshcilvwl/flzkvrshbq/ReadVariableOp_2ReadVariableOp/owshcilvwl_flzkvrshbq_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&owshcilvwl/flzkvrshbq/ReadVariableOp_2Ä
owshcilvwl/flzkvrshbq/mul_4Mul.owshcilvwl/flzkvrshbq/ReadVariableOp_2:value:0owshcilvwl/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mul_4¼
owshcilvwl/flzkvrshbq/add_4AddV2$owshcilvwl/flzkvrshbq/split:output:3owshcilvwl/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/add_4 
owshcilvwl/flzkvrshbq/Sigmoid_2Sigmoidowshcilvwl/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
owshcilvwl/flzkvrshbq/Sigmoid_2
owshcilvwl/flzkvrshbq/Tanh_1Tanhowshcilvwl/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/Tanh_1º
owshcilvwl/flzkvrshbq/mul_5Mul#owshcilvwl/flzkvrshbq/Sigmoid_2:y:0 owshcilvwl/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mul_5¥
(owshcilvwl/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(owshcilvwl/TensorArrayV2_1/element_shapeä
owshcilvwl/TensorArrayV2_1TensorListReserve1owshcilvwl/TensorArrayV2_1/element_shape:output:0#owshcilvwl/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
owshcilvwl/TensorArrayV2_1d
owshcilvwl/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/time
#owshcilvwl/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#owshcilvwl/while/maximum_iterations
owshcilvwl/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/while/loop_counter°
owshcilvwl/whileWhile&owshcilvwl/while/loop_counter:output:0,owshcilvwl/while/maximum_iterations:output:0owshcilvwl/time:output:0#owshcilvwl/TensorArrayV2_1:handle:0owshcilvwl/zeros:output:0owshcilvwl/zeros_1:output:0#owshcilvwl/strided_slice_1:output:0Bowshcilvwl/TensorArrayUnstack/TensorListFromTensor:output_handle:04owshcilvwl_flzkvrshbq_matmul_readvariableop_resource6owshcilvwl_flzkvrshbq_matmul_1_readvariableop_resource5owshcilvwl_flzkvrshbq_biasadd_readvariableop_resource-owshcilvwl_flzkvrshbq_readvariableop_resource/owshcilvwl_flzkvrshbq_readvariableop_1_resource/owshcilvwl_flzkvrshbq_readvariableop_2_resource*
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
owshcilvwl_while_body_657075*(
cond R
owshcilvwl_while_cond_657074*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
owshcilvwl/whileË
;owshcilvwl/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;owshcilvwl/TensorArrayV2Stack/TensorListStack/element_shape
-owshcilvwl/TensorArrayV2Stack/TensorListStackTensorListStackowshcilvwl/while:output:3Dowshcilvwl/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-owshcilvwl/TensorArrayV2Stack/TensorListStack
 owshcilvwl/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 owshcilvwl/strided_slice_3/stack
"owshcilvwl/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"owshcilvwl/strided_slice_3/stack_1
"owshcilvwl/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"owshcilvwl/strided_slice_3/stack_2Ü
owshcilvwl/strided_slice_3StridedSlice6owshcilvwl/TensorArrayV2Stack/TensorListStack:tensor:0)owshcilvwl/strided_slice_3/stack:output:0+owshcilvwl/strided_slice_3/stack_1:output:0+owshcilvwl/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
owshcilvwl/strided_slice_3
owshcilvwl/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
owshcilvwl/transpose_1/permÑ
owshcilvwl/transpose_1	Transpose6owshcilvwl/TensorArrayV2Stack/TensorListStack:tensor:0$owshcilvwl/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/transpose_1®
 oaettnoaty/MatMul/ReadVariableOpReadVariableOp)oaettnoaty_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 oaettnoaty/MatMul/ReadVariableOp±
oaettnoaty/MatMulMatMul#owshcilvwl/strided_slice_3:output:0(oaettnoaty/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oaettnoaty/MatMul­
!oaettnoaty/BiasAdd/ReadVariableOpReadVariableOp*oaettnoaty_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!oaettnoaty/BiasAdd/ReadVariableOp­
oaettnoaty/BiasAddBiasAddoaettnoaty/MatMul:product:0)oaettnoaty/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oaettnoaty/BiasAddÏ
IdentityIdentityoaettnoaty/BiasAdd:output:0.^bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp5^bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp"^oaettnoaty/BiasAdd/ReadVariableOp!^oaettnoaty/MatMul/ReadVariableOp-^osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp,^osutmzfngz/dsycfvoega/MatMul/ReadVariableOp.^osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp%^osutmzfngz/dsycfvoega/ReadVariableOp'^osutmzfngz/dsycfvoega/ReadVariableOp_1'^osutmzfngz/dsycfvoega/ReadVariableOp_2^osutmzfngz/while-^owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp,^owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp.^owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp%^owshcilvwl/flzkvrshbq/ReadVariableOp'^owshcilvwl/flzkvrshbq/ReadVariableOp_1'^owshcilvwl/flzkvrshbq/ReadVariableOp_2^owshcilvwl/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2^
-bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp-bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp2l
4bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp4bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!oaettnoaty/BiasAdd/ReadVariableOp!oaettnoaty/BiasAdd/ReadVariableOp2D
 oaettnoaty/MatMul/ReadVariableOp oaettnoaty/MatMul/ReadVariableOp2\
,osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp,osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp2Z
+osutmzfngz/dsycfvoega/MatMul/ReadVariableOp+osutmzfngz/dsycfvoega/MatMul/ReadVariableOp2^
-osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp-osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp2L
$osutmzfngz/dsycfvoega/ReadVariableOp$osutmzfngz/dsycfvoega/ReadVariableOp2P
&osutmzfngz/dsycfvoega/ReadVariableOp_1&osutmzfngz/dsycfvoega/ReadVariableOp_12P
&osutmzfngz/dsycfvoega/ReadVariableOp_2&osutmzfngz/dsycfvoega/ReadVariableOp_22$
osutmzfngz/whileosutmzfngz/while2\
,owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp,owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp2Z
+owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp+owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp2^
-owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp-owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp2L
$owshcilvwl/flzkvrshbq/ReadVariableOp$owshcilvwl/flzkvrshbq/ReadVariableOp2P
&owshcilvwl/flzkvrshbq/ReadVariableOp_1&owshcilvwl/flzkvrshbq/ReadVariableOp_12P
&owshcilvwl/flzkvrshbq/ReadVariableOp_2&owshcilvwl/flzkvrshbq/ReadVariableOp_22$
owshcilvwl/whileowshcilvwl/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


å
while_cond_655803
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_655803___redundant_placeholder04
0while_while_cond_655803___redundant_placeholder14
0while_while_cond_655803___redundant_placeholder24
0while_while_cond_655803___redundant_placeholder34
0while_while_cond_655803___redundant_placeholder44
0while_while_cond_655803___redundant_placeholder54
0while_while_cond_655803___redundant_placeholder6
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
+__inference_dsycfvoega_layer_call_fn_659291

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
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_6542262
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
Ñ

+__inference_owshcilvwl_layer_call_fn_658489

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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_6559052
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
°'
²
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_654797

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
p
É
osutmzfngz_while_body_6573032
.osutmzfngz_while_osutmzfngz_while_loop_counter8
4osutmzfngz_while_osutmzfngz_while_maximum_iterations 
osutmzfngz_while_placeholder"
osutmzfngz_while_placeholder_1"
osutmzfngz_while_placeholder_2"
osutmzfngz_while_placeholder_31
-osutmzfngz_while_osutmzfngz_strided_slice_1_0m
iosutmzfngz_while_tensorarrayv2read_tensorlistgetitem_osutmzfngz_tensorarrayunstack_tensorlistfromtensor_0O
<osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource_0:	Q
>osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource_0:	 L
=osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource_0:	C
5osutmzfngz_while_dsycfvoega_readvariableop_resource_0: E
7osutmzfngz_while_dsycfvoega_readvariableop_1_resource_0: E
7osutmzfngz_while_dsycfvoega_readvariableop_2_resource_0: 
osutmzfngz_while_identity
osutmzfngz_while_identity_1
osutmzfngz_while_identity_2
osutmzfngz_while_identity_3
osutmzfngz_while_identity_4
osutmzfngz_while_identity_5/
+osutmzfngz_while_osutmzfngz_strided_slice_1k
gosutmzfngz_while_tensorarrayv2read_tensorlistgetitem_osutmzfngz_tensorarrayunstack_tensorlistfromtensorM
:osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource:	O
<osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource:	 J
;osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource:	A
3osutmzfngz_while_dsycfvoega_readvariableop_resource: C
5osutmzfngz_while_dsycfvoega_readvariableop_1_resource: C
5osutmzfngz_while_dsycfvoega_readvariableop_2_resource: ¢2osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp¢1osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp¢3osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp¢*osutmzfngz/while/dsycfvoega/ReadVariableOp¢,osutmzfngz/while/dsycfvoega/ReadVariableOp_1¢,osutmzfngz/while/dsycfvoega/ReadVariableOp_2Ù
Bosutmzfngz/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bosutmzfngz/while/TensorArrayV2Read/TensorListGetItem/element_shape
4osutmzfngz/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiosutmzfngz_while_tensorarrayv2read_tensorlistgetitem_osutmzfngz_tensorarrayunstack_tensorlistfromtensor_0osutmzfngz_while_placeholderKosutmzfngz/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4osutmzfngz/while/TensorArrayV2Read/TensorListGetItemä
1osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOpReadVariableOp<osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOpý
"osutmzfngz/while/dsycfvoega/MatMulMatMul;osutmzfngz/while/TensorArrayV2Read/TensorListGetItem:item:09osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"osutmzfngz/while/dsycfvoega/MatMulê
3osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp>osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOpæ
$osutmzfngz/while/dsycfvoega/MatMul_1MatMulosutmzfngz_while_placeholder_2;osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$osutmzfngz/while/dsycfvoega/MatMul_1Ü
osutmzfngz/while/dsycfvoega/addAddV2,osutmzfngz/while/dsycfvoega/MatMul:product:0.osutmzfngz/while/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
osutmzfngz/while/dsycfvoega/addã
2osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp=osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOpé
#osutmzfngz/while/dsycfvoega/BiasAddBiasAdd#osutmzfngz/while/dsycfvoega/add:z:0:osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#osutmzfngz/while/dsycfvoega/BiasAdd
+osutmzfngz/while/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+osutmzfngz/while/dsycfvoega/split/split_dim¯
!osutmzfngz/while/dsycfvoega/splitSplit4osutmzfngz/while/dsycfvoega/split/split_dim:output:0,osutmzfngz/while/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!osutmzfngz/while/dsycfvoega/splitÊ
*osutmzfngz/while/dsycfvoega/ReadVariableOpReadVariableOp5osutmzfngz_while_dsycfvoega_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*osutmzfngz/while/dsycfvoega/ReadVariableOpÏ
osutmzfngz/while/dsycfvoega/mulMul2osutmzfngz/while/dsycfvoega/ReadVariableOp:value:0osutmzfngz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
osutmzfngz/while/dsycfvoega/mulÒ
!osutmzfngz/while/dsycfvoega/add_1AddV2*osutmzfngz/while/dsycfvoega/split:output:0#osutmzfngz/while/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/add_1®
#osutmzfngz/while/dsycfvoega/SigmoidSigmoid%osutmzfngz/while/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#osutmzfngz/while/dsycfvoega/SigmoidÐ
,osutmzfngz/while/dsycfvoega/ReadVariableOp_1ReadVariableOp7osutmzfngz_while_dsycfvoega_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,osutmzfngz/while/dsycfvoega/ReadVariableOp_1Õ
!osutmzfngz/while/dsycfvoega/mul_1Mul4osutmzfngz/while/dsycfvoega/ReadVariableOp_1:value:0osutmzfngz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/mul_1Ô
!osutmzfngz/while/dsycfvoega/add_2AddV2*osutmzfngz/while/dsycfvoega/split:output:1%osutmzfngz/while/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/add_2²
%osutmzfngz/while/dsycfvoega/Sigmoid_1Sigmoid%osutmzfngz/while/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%osutmzfngz/while/dsycfvoega/Sigmoid_1Ê
!osutmzfngz/while/dsycfvoega/mul_2Mul)osutmzfngz/while/dsycfvoega/Sigmoid_1:y:0osutmzfngz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/mul_2ª
 osutmzfngz/while/dsycfvoega/TanhTanh*osutmzfngz/while/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 osutmzfngz/while/dsycfvoega/TanhÎ
!osutmzfngz/while/dsycfvoega/mul_3Mul'osutmzfngz/while/dsycfvoega/Sigmoid:y:0$osutmzfngz/while/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/mul_3Ï
!osutmzfngz/while/dsycfvoega/add_3AddV2%osutmzfngz/while/dsycfvoega/mul_2:z:0%osutmzfngz/while/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/add_3Ð
,osutmzfngz/while/dsycfvoega/ReadVariableOp_2ReadVariableOp7osutmzfngz_while_dsycfvoega_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,osutmzfngz/while/dsycfvoega/ReadVariableOp_2Ü
!osutmzfngz/while/dsycfvoega/mul_4Mul4osutmzfngz/while/dsycfvoega/ReadVariableOp_2:value:0%osutmzfngz/while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/mul_4Ô
!osutmzfngz/while/dsycfvoega/add_4AddV2*osutmzfngz/while/dsycfvoega/split:output:3%osutmzfngz/while/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/add_4²
%osutmzfngz/while/dsycfvoega/Sigmoid_2Sigmoid%osutmzfngz/while/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%osutmzfngz/while/dsycfvoega/Sigmoid_2©
"osutmzfngz/while/dsycfvoega/Tanh_1Tanh%osutmzfngz/while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"osutmzfngz/while/dsycfvoega/Tanh_1Ò
!osutmzfngz/while/dsycfvoega/mul_5Mul)osutmzfngz/while/dsycfvoega/Sigmoid_2:y:0&osutmzfngz/while/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/mul_5
5osutmzfngz/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemosutmzfngz_while_placeholder_1osutmzfngz_while_placeholder%osutmzfngz/while/dsycfvoega/mul_5:z:0*
_output_shapes
: *
element_dtype027
5osutmzfngz/while/TensorArrayV2Write/TensorListSetItemr
osutmzfngz/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
osutmzfngz/while/add/y
osutmzfngz/while/addAddV2osutmzfngz_while_placeholderosutmzfngz/while/add/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/while/addv
osutmzfngz/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
osutmzfngz/while/add_1/y­
osutmzfngz/while/add_1AddV2.osutmzfngz_while_osutmzfngz_while_loop_counter!osutmzfngz/while/add_1/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/while/add_1©
osutmzfngz/while/IdentityIdentityosutmzfngz/while/add_1:z:03^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
osutmzfngz/while/IdentityÇ
osutmzfngz/while/Identity_1Identity4osutmzfngz_while_osutmzfngz_while_maximum_iterations3^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
osutmzfngz/while/Identity_1«
osutmzfngz/while/Identity_2Identityosutmzfngz/while/add:z:03^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
osutmzfngz/while/Identity_2Ø
osutmzfngz/while/Identity_3IdentityEosutmzfngz/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
osutmzfngz/while/Identity_3É
osutmzfngz/while/Identity_4Identity%osutmzfngz/while/dsycfvoega/mul_5:z:03^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/while/Identity_4É
osutmzfngz/while/Identity_5Identity%osutmzfngz/while/dsycfvoega/add_3:z:03^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/while/Identity_5"|
;osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource=osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource_0"~
<osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource>osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource_0"z
:osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource<osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource_0"p
5osutmzfngz_while_dsycfvoega_readvariableop_1_resource7osutmzfngz_while_dsycfvoega_readvariableop_1_resource_0"p
5osutmzfngz_while_dsycfvoega_readvariableop_2_resource7osutmzfngz_while_dsycfvoega_readvariableop_2_resource_0"l
3osutmzfngz_while_dsycfvoega_readvariableop_resource5osutmzfngz_while_dsycfvoega_readvariableop_resource_0"?
osutmzfngz_while_identity"osutmzfngz/while/Identity:output:0"C
osutmzfngz_while_identity_1$osutmzfngz/while/Identity_1:output:0"C
osutmzfngz_while_identity_2$osutmzfngz/while/Identity_2:output:0"C
osutmzfngz_while_identity_3$osutmzfngz/while/Identity_3:output:0"C
osutmzfngz_while_identity_4$osutmzfngz/while/Identity_4:output:0"C
osutmzfngz_while_identity_5$osutmzfngz/while/Identity_5:output:0"\
+osutmzfngz_while_osutmzfngz_strided_slice_1-osutmzfngz_while_osutmzfngz_strided_slice_1_0"Ô
gosutmzfngz_while_tensorarrayv2read_tensorlistgetitem_osutmzfngz_tensorarrayunstack_tensorlistfromtensoriosutmzfngz_while_tensorarrayv2read_tensorlistgetitem_osutmzfngz_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2f
1osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp1osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp2j
3osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp3osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp2X
*osutmzfngz/while/dsycfvoega/ReadVariableOp*osutmzfngz/while/dsycfvoega/ReadVariableOp2\
,osutmzfngz/while/dsycfvoega/ReadVariableOp_1,osutmzfngz/while/dsycfvoega/ReadVariableOp_12\
,osutmzfngz/while/dsycfvoega/ReadVariableOp_2,osutmzfngz/while/dsycfvoega/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_654059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_dsycfvoega_654083_0:	,
while_dsycfvoega_654085_0:	 (
while_dsycfvoega_654087_0:	'
while_dsycfvoega_654089_0: '
while_dsycfvoega_654091_0: '
while_dsycfvoega_654093_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_dsycfvoega_654083:	*
while_dsycfvoega_654085:	 &
while_dsycfvoega_654087:	%
while_dsycfvoega_654089: %
while_dsycfvoega_654091: %
while_dsycfvoega_654093: ¢(while/dsycfvoega/StatefulPartitionedCallÃ
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
(while/dsycfvoega/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_dsycfvoega_654083_0while_dsycfvoega_654085_0while_dsycfvoega_654087_0while_dsycfvoega_654089_0while_dsycfvoega_654091_0while_dsycfvoega_654093_0*
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
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_6540392*
(while/dsycfvoega/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/dsycfvoega/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/dsycfvoega/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/dsycfvoega/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/dsycfvoega/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/dsycfvoega/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/dsycfvoega/StatefulPartitionedCall:output:1)^while/dsycfvoega/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/dsycfvoega/StatefulPartitionedCall:output:2)^while/dsycfvoega/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_dsycfvoega_654083while_dsycfvoega_654083_0"4
while_dsycfvoega_654085while_dsycfvoega_654085_0"4
while_dsycfvoega_654087while_dsycfvoega_654087_0"4
while_dsycfvoega_654089while_dsycfvoega_654089_0"4
while_dsycfvoega_654091while_dsycfvoega_654091_0"4
while_dsycfvoega_654093while_dsycfvoega_654093_0")
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
(while/dsycfvoega/StatefulPartitionedCall(while/dsycfvoega/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_654984

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
ê

$__inference_signature_wrapper_656704

jfowsgvbzw
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
jfowsgvbzwunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_6539522
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
jfowsgvbzw

b
F__inference_xlcvyoxoxq_layer_call_and_return_conditional_losses_655531

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
¨0
»
F__inference_bpstkcuudk_layer_call_and_return_conditional_losses_657632

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
¨0
»
F__inference_bpstkcuudk_layer_call_and_return_conditional_losses_655512

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
p
É
osutmzfngz_while_body_6568992
.osutmzfngz_while_osutmzfngz_while_loop_counter8
4osutmzfngz_while_osutmzfngz_while_maximum_iterations 
osutmzfngz_while_placeholder"
osutmzfngz_while_placeholder_1"
osutmzfngz_while_placeholder_2"
osutmzfngz_while_placeholder_31
-osutmzfngz_while_osutmzfngz_strided_slice_1_0m
iosutmzfngz_while_tensorarrayv2read_tensorlistgetitem_osutmzfngz_tensorarrayunstack_tensorlistfromtensor_0O
<osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource_0:	Q
>osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource_0:	 L
=osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource_0:	C
5osutmzfngz_while_dsycfvoega_readvariableop_resource_0: E
7osutmzfngz_while_dsycfvoega_readvariableop_1_resource_0: E
7osutmzfngz_while_dsycfvoega_readvariableop_2_resource_0: 
osutmzfngz_while_identity
osutmzfngz_while_identity_1
osutmzfngz_while_identity_2
osutmzfngz_while_identity_3
osutmzfngz_while_identity_4
osutmzfngz_while_identity_5/
+osutmzfngz_while_osutmzfngz_strided_slice_1k
gosutmzfngz_while_tensorarrayv2read_tensorlistgetitem_osutmzfngz_tensorarrayunstack_tensorlistfromtensorM
:osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource:	O
<osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource:	 J
;osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource:	A
3osutmzfngz_while_dsycfvoega_readvariableop_resource: C
5osutmzfngz_while_dsycfvoega_readvariableop_1_resource: C
5osutmzfngz_while_dsycfvoega_readvariableop_2_resource: ¢2osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp¢1osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp¢3osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp¢*osutmzfngz/while/dsycfvoega/ReadVariableOp¢,osutmzfngz/while/dsycfvoega/ReadVariableOp_1¢,osutmzfngz/while/dsycfvoega/ReadVariableOp_2Ù
Bosutmzfngz/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bosutmzfngz/while/TensorArrayV2Read/TensorListGetItem/element_shape
4osutmzfngz/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiosutmzfngz_while_tensorarrayv2read_tensorlistgetitem_osutmzfngz_tensorarrayunstack_tensorlistfromtensor_0osutmzfngz_while_placeholderKosutmzfngz/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4osutmzfngz/while/TensorArrayV2Read/TensorListGetItemä
1osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOpReadVariableOp<osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOpý
"osutmzfngz/while/dsycfvoega/MatMulMatMul;osutmzfngz/while/TensorArrayV2Read/TensorListGetItem:item:09osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"osutmzfngz/while/dsycfvoega/MatMulê
3osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp>osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOpæ
$osutmzfngz/while/dsycfvoega/MatMul_1MatMulosutmzfngz_while_placeholder_2;osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$osutmzfngz/while/dsycfvoega/MatMul_1Ü
osutmzfngz/while/dsycfvoega/addAddV2,osutmzfngz/while/dsycfvoega/MatMul:product:0.osutmzfngz/while/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
osutmzfngz/while/dsycfvoega/addã
2osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp=osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOpé
#osutmzfngz/while/dsycfvoega/BiasAddBiasAdd#osutmzfngz/while/dsycfvoega/add:z:0:osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#osutmzfngz/while/dsycfvoega/BiasAdd
+osutmzfngz/while/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+osutmzfngz/while/dsycfvoega/split/split_dim¯
!osutmzfngz/while/dsycfvoega/splitSplit4osutmzfngz/while/dsycfvoega/split/split_dim:output:0,osutmzfngz/while/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!osutmzfngz/while/dsycfvoega/splitÊ
*osutmzfngz/while/dsycfvoega/ReadVariableOpReadVariableOp5osutmzfngz_while_dsycfvoega_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*osutmzfngz/while/dsycfvoega/ReadVariableOpÏ
osutmzfngz/while/dsycfvoega/mulMul2osutmzfngz/while/dsycfvoega/ReadVariableOp:value:0osutmzfngz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
osutmzfngz/while/dsycfvoega/mulÒ
!osutmzfngz/while/dsycfvoega/add_1AddV2*osutmzfngz/while/dsycfvoega/split:output:0#osutmzfngz/while/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/add_1®
#osutmzfngz/while/dsycfvoega/SigmoidSigmoid%osutmzfngz/while/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#osutmzfngz/while/dsycfvoega/SigmoidÐ
,osutmzfngz/while/dsycfvoega/ReadVariableOp_1ReadVariableOp7osutmzfngz_while_dsycfvoega_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,osutmzfngz/while/dsycfvoega/ReadVariableOp_1Õ
!osutmzfngz/while/dsycfvoega/mul_1Mul4osutmzfngz/while/dsycfvoega/ReadVariableOp_1:value:0osutmzfngz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/mul_1Ô
!osutmzfngz/while/dsycfvoega/add_2AddV2*osutmzfngz/while/dsycfvoega/split:output:1%osutmzfngz/while/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/add_2²
%osutmzfngz/while/dsycfvoega/Sigmoid_1Sigmoid%osutmzfngz/while/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%osutmzfngz/while/dsycfvoega/Sigmoid_1Ê
!osutmzfngz/while/dsycfvoega/mul_2Mul)osutmzfngz/while/dsycfvoega/Sigmoid_1:y:0osutmzfngz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/mul_2ª
 osutmzfngz/while/dsycfvoega/TanhTanh*osutmzfngz/while/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 osutmzfngz/while/dsycfvoega/TanhÎ
!osutmzfngz/while/dsycfvoega/mul_3Mul'osutmzfngz/while/dsycfvoega/Sigmoid:y:0$osutmzfngz/while/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/mul_3Ï
!osutmzfngz/while/dsycfvoega/add_3AddV2%osutmzfngz/while/dsycfvoega/mul_2:z:0%osutmzfngz/while/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/add_3Ð
,osutmzfngz/while/dsycfvoega/ReadVariableOp_2ReadVariableOp7osutmzfngz_while_dsycfvoega_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,osutmzfngz/while/dsycfvoega/ReadVariableOp_2Ü
!osutmzfngz/while/dsycfvoega/mul_4Mul4osutmzfngz/while/dsycfvoega/ReadVariableOp_2:value:0%osutmzfngz/while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/mul_4Ô
!osutmzfngz/while/dsycfvoega/add_4AddV2*osutmzfngz/while/dsycfvoega/split:output:3%osutmzfngz/while/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/add_4²
%osutmzfngz/while/dsycfvoega/Sigmoid_2Sigmoid%osutmzfngz/while/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%osutmzfngz/while/dsycfvoega/Sigmoid_2©
"osutmzfngz/while/dsycfvoega/Tanh_1Tanh%osutmzfngz/while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"osutmzfngz/while/dsycfvoega/Tanh_1Ò
!osutmzfngz/while/dsycfvoega/mul_5Mul)osutmzfngz/while/dsycfvoega/Sigmoid_2:y:0&osutmzfngz/while/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!osutmzfngz/while/dsycfvoega/mul_5
5osutmzfngz/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemosutmzfngz_while_placeholder_1osutmzfngz_while_placeholder%osutmzfngz/while/dsycfvoega/mul_5:z:0*
_output_shapes
: *
element_dtype027
5osutmzfngz/while/TensorArrayV2Write/TensorListSetItemr
osutmzfngz/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
osutmzfngz/while/add/y
osutmzfngz/while/addAddV2osutmzfngz_while_placeholderosutmzfngz/while/add/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/while/addv
osutmzfngz/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
osutmzfngz/while/add_1/y­
osutmzfngz/while/add_1AddV2.osutmzfngz_while_osutmzfngz_while_loop_counter!osutmzfngz/while/add_1/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/while/add_1©
osutmzfngz/while/IdentityIdentityosutmzfngz/while/add_1:z:03^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
osutmzfngz/while/IdentityÇ
osutmzfngz/while/Identity_1Identity4osutmzfngz_while_osutmzfngz_while_maximum_iterations3^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
osutmzfngz/while/Identity_1«
osutmzfngz/while/Identity_2Identityosutmzfngz/while/add:z:03^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
osutmzfngz/while/Identity_2Ø
osutmzfngz/while/Identity_3IdentityEosutmzfngz/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
osutmzfngz/while/Identity_3É
osutmzfngz/while/Identity_4Identity%osutmzfngz/while/dsycfvoega/mul_5:z:03^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/while/Identity_4É
osutmzfngz/while/Identity_5Identity%osutmzfngz/while/dsycfvoega/add_3:z:03^osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2^osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp4^osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp+^osutmzfngz/while/dsycfvoega/ReadVariableOp-^osutmzfngz/while/dsycfvoega/ReadVariableOp_1-^osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/while/Identity_5"|
;osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource=osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource_0"~
<osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource>osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource_0"z
:osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource<osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource_0"p
5osutmzfngz_while_dsycfvoega_readvariableop_1_resource7osutmzfngz_while_dsycfvoega_readvariableop_1_resource_0"p
5osutmzfngz_while_dsycfvoega_readvariableop_2_resource7osutmzfngz_while_dsycfvoega_readvariableop_2_resource_0"l
3osutmzfngz_while_dsycfvoega_readvariableop_resource5osutmzfngz_while_dsycfvoega_readvariableop_resource_0"?
osutmzfngz_while_identity"osutmzfngz/while/Identity:output:0"C
osutmzfngz_while_identity_1$osutmzfngz/while/Identity_1:output:0"C
osutmzfngz_while_identity_2$osutmzfngz/while/Identity_2:output:0"C
osutmzfngz_while_identity_3$osutmzfngz/while/Identity_3:output:0"C
osutmzfngz_while_identity_4$osutmzfngz/while/Identity_4:output:0"C
osutmzfngz_while_identity_5$osutmzfngz/while/Identity_5:output:0"\
+osutmzfngz_while_osutmzfngz_strided_slice_1-osutmzfngz_while_osutmzfngz_strided_slice_1_0"Ô
gosutmzfngz_while_tensorarrayv2read_tensorlistgetitem_osutmzfngz_tensorarrayunstack_tensorlistfromtensoriosutmzfngz_while_tensorarrayv2read_tensorlistgetitem_osutmzfngz_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2f
1osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp1osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp2j
3osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp3osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp2X
*osutmzfngz/while/dsycfvoega/ReadVariableOp*osutmzfngz/while/dsycfvoega/ReadVariableOp2\
,osutmzfngz/while/dsycfvoega/ReadVariableOp_1,osutmzfngz/while/dsycfvoega/ReadVariableOp_12\
,osutmzfngz/while/dsycfvoega/ReadVariableOp_2,osutmzfngz/while/dsycfvoega/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
Ýh

F__inference_osutmzfngz_layer_call_and_return_conditional_losses_658078
inputs_0<
)dsycfvoega_matmul_readvariableop_resource:	>
+dsycfvoega_matmul_1_readvariableop_resource:	 9
*dsycfvoega_biasadd_readvariableop_resource:	0
"dsycfvoega_readvariableop_resource: 2
$dsycfvoega_readvariableop_1_resource: 2
$dsycfvoega_readvariableop_2_resource: 
identity¢!dsycfvoega/BiasAdd/ReadVariableOp¢ dsycfvoega/MatMul/ReadVariableOp¢"dsycfvoega/MatMul_1/ReadVariableOp¢dsycfvoega/ReadVariableOp¢dsycfvoega/ReadVariableOp_1¢dsycfvoega/ReadVariableOp_2¢whileF
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
 dsycfvoega/MatMul/ReadVariableOpReadVariableOp)dsycfvoega_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dsycfvoega/MatMul/ReadVariableOp§
dsycfvoega/MatMulMatMulstrided_slice_2:output:0(dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMulµ
"dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp+dsycfvoega_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dsycfvoega/MatMul_1/ReadVariableOp£
dsycfvoega/MatMul_1MatMulzeros:output:0*dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMul_1
dsycfvoega/addAddV2dsycfvoega/MatMul:product:0dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/add®
!dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp*dsycfvoega_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dsycfvoega/BiasAdd/ReadVariableOp¥
dsycfvoega/BiasAddBiasAdddsycfvoega/add:z:0)dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/BiasAddz
dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dsycfvoega/split/split_dimë
dsycfvoega/splitSplit#dsycfvoega/split/split_dim:output:0dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dsycfvoega/split
dsycfvoega/ReadVariableOpReadVariableOp"dsycfvoega_readvariableop_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp
dsycfvoega/mulMul!dsycfvoega/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul
dsycfvoega/add_1AddV2dsycfvoega/split:output:0dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_1{
dsycfvoega/SigmoidSigmoiddsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid
dsycfvoega/ReadVariableOp_1ReadVariableOp$dsycfvoega_readvariableop_1_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_1
dsycfvoega/mul_1Mul#dsycfvoega/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_1
dsycfvoega/add_2AddV2dsycfvoega/split:output:1dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_2
dsycfvoega/Sigmoid_1Sigmoiddsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_1
dsycfvoega/mul_2Muldsycfvoega/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_2w
dsycfvoega/TanhTanhdsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh
dsycfvoega/mul_3Muldsycfvoega/Sigmoid:y:0dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_3
dsycfvoega/add_3AddV2dsycfvoega/mul_2:z:0dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_3
dsycfvoega/ReadVariableOp_2ReadVariableOp$dsycfvoega_readvariableop_2_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_2
dsycfvoega/mul_4Mul#dsycfvoega/ReadVariableOp_2:value:0dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_4
dsycfvoega/add_4AddV2dsycfvoega/split:output:3dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_4
dsycfvoega/Sigmoid_2Sigmoiddsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_2v
dsycfvoega/Tanh_1Tanhdsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh_1
dsycfvoega/mul_5Muldsycfvoega/Sigmoid_2:y:0dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dsycfvoega_matmul_readvariableop_resource+dsycfvoega_matmul_1_readvariableop_resource*dsycfvoega_biasadd_readvariableop_resource"dsycfvoega_readvariableop_resource$dsycfvoega_readvariableop_1_resource$dsycfvoega_readvariableop_2_resource*
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
while_body_657977*
condR
while_cond_657976*Q
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
IdentityIdentitytranspose_1:y:0"^dsycfvoega/BiasAdd/ReadVariableOp!^dsycfvoega/MatMul/ReadVariableOp#^dsycfvoega/MatMul_1/ReadVariableOp^dsycfvoega/ReadVariableOp^dsycfvoega/ReadVariableOp_1^dsycfvoega/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dsycfvoega/BiasAdd/ReadVariableOp!dsycfvoega/BiasAdd/ReadVariableOp2D
 dsycfvoega/MatMul/ReadVariableOp dsycfvoega/MatMul/ReadVariableOp2H
"dsycfvoega/MatMul_1/ReadVariableOp"dsycfvoega/MatMul_1/ReadVariableOp26
dsycfvoega/ReadVariableOpdsycfvoega/ReadVariableOp2:
dsycfvoega/ReadVariableOp_1dsycfvoega/ReadVariableOp_12:
dsycfvoega/ReadVariableOp_2dsycfvoega/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
h

F__inference_osutmzfngz_layer_call_and_return_conditional_losses_658438

inputs<
)dsycfvoega_matmul_readvariableop_resource:	>
+dsycfvoega_matmul_1_readvariableop_resource:	 9
*dsycfvoega_biasadd_readvariableop_resource:	0
"dsycfvoega_readvariableop_resource: 2
$dsycfvoega_readvariableop_1_resource: 2
$dsycfvoega_readvariableop_2_resource: 
identity¢!dsycfvoega/BiasAdd/ReadVariableOp¢ dsycfvoega/MatMul/ReadVariableOp¢"dsycfvoega/MatMul_1/ReadVariableOp¢dsycfvoega/ReadVariableOp¢dsycfvoega/ReadVariableOp_1¢dsycfvoega/ReadVariableOp_2¢whileD
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
 dsycfvoega/MatMul/ReadVariableOpReadVariableOp)dsycfvoega_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dsycfvoega/MatMul/ReadVariableOp§
dsycfvoega/MatMulMatMulstrided_slice_2:output:0(dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMulµ
"dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp+dsycfvoega_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dsycfvoega/MatMul_1/ReadVariableOp£
dsycfvoega/MatMul_1MatMulzeros:output:0*dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMul_1
dsycfvoega/addAddV2dsycfvoega/MatMul:product:0dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/add®
!dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp*dsycfvoega_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dsycfvoega/BiasAdd/ReadVariableOp¥
dsycfvoega/BiasAddBiasAdddsycfvoega/add:z:0)dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/BiasAddz
dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dsycfvoega/split/split_dimë
dsycfvoega/splitSplit#dsycfvoega/split/split_dim:output:0dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dsycfvoega/split
dsycfvoega/ReadVariableOpReadVariableOp"dsycfvoega_readvariableop_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp
dsycfvoega/mulMul!dsycfvoega/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul
dsycfvoega/add_1AddV2dsycfvoega/split:output:0dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_1{
dsycfvoega/SigmoidSigmoiddsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid
dsycfvoega/ReadVariableOp_1ReadVariableOp$dsycfvoega_readvariableop_1_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_1
dsycfvoega/mul_1Mul#dsycfvoega/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_1
dsycfvoega/add_2AddV2dsycfvoega/split:output:1dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_2
dsycfvoega/Sigmoid_1Sigmoiddsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_1
dsycfvoega/mul_2Muldsycfvoega/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_2w
dsycfvoega/TanhTanhdsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh
dsycfvoega/mul_3Muldsycfvoega/Sigmoid:y:0dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_3
dsycfvoega/add_3AddV2dsycfvoega/mul_2:z:0dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_3
dsycfvoega/ReadVariableOp_2ReadVariableOp$dsycfvoega_readvariableop_2_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_2
dsycfvoega/mul_4Mul#dsycfvoega/ReadVariableOp_2:value:0dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_4
dsycfvoega/add_4AddV2dsycfvoega/split:output:3dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_4
dsycfvoega/Sigmoid_2Sigmoiddsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_2v
dsycfvoega/Tanh_1Tanhdsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh_1
dsycfvoega/mul_5Muldsycfvoega/Sigmoid_2:y:0dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dsycfvoega_matmul_readvariableop_resource+dsycfvoega_matmul_1_readvariableop_resource*dsycfvoega_biasadd_readvariableop_resource"dsycfvoega_readvariableop_resource$dsycfvoega_readvariableop_1_resource$dsycfvoega_readvariableop_2_resource*
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
while_body_658337*
condR
while_cond_658336*Q
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
IdentityIdentitytranspose_1:y:0"^dsycfvoega/BiasAdd/ReadVariableOp!^dsycfvoega/MatMul/ReadVariableOp#^dsycfvoega/MatMul_1/ReadVariableOp^dsycfvoega/ReadVariableOp^dsycfvoega/ReadVariableOp_1^dsycfvoega/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dsycfvoega/BiasAdd/ReadVariableOp!dsycfvoega/BiasAdd/ReadVariableOp2D
 dsycfvoega/MatMul/ReadVariableOp dsycfvoega/MatMul/ReadVariableOp2H
"dsycfvoega/MatMul_1/ReadVariableOp"dsycfvoega/MatMul_1/ReadVariableOp26
dsycfvoega/ReadVariableOpdsycfvoega/ReadVariableOp2:
dsycfvoega/ReadVariableOp_1dsycfvoega/ReadVariableOp_12:
dsycfvoega/ReadVariableOp_2dsycfvoega/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
F
ã
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_654897

inputs$
flzkvrshbq_654798:	 $
flzkvrshbq_654800:	  
flzkvrshbq_654802:	
flzkvrshbq_654804: 
flzkvrshbq_654806: 
flzkvrshbq_654808: 
identity¢"flzkvrshbq/StatefulPartitionedCall¢whileD
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
"flzkvrshbq/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0flzkvrshbq_654798flzkvrshbq_654800flzkvrshbq_654802flzkvrshbq_654804flzkvrshbq_654806flzkvrshbq_654808*
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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_6547972$
"flzkvrshbq/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0flzkvrshbq_654798flzkvrshbq_654800flzkvrshbq_654802flzkvrshbq_654804flzkvrshbq_654806flzkvrshbq_654808*
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
while_body_654817*
condR
while_cond_654816*Q
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
IdentityIdentitystrided_slice_3:output:0#^flzkvrshbq/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"flzkvrshbq/StatefulPartitionedCall"flzkvrshbq/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ßY

while_body_657977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dsycfvoega_matmul_readvariableop_resource_0:	F
3while_dsycfvoega_matmul_1_readvariableop_resource_0:	 A
2while_dsycfvoega_biasadd_readvariableop_resource_0:	8
*while_dsycfvoega_readvariableop_resource_0: :
,while_dsycfvoega_readvariableop_1_resource_0: :
,while_dsycfvoega_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dsycfvoega_matmul_readvariableop_resource:	D
1while_dsycfvoega_matmul_1_readvariableop_resource:	 ?
0while_dsycfvoega_biasadd_readvariableop_resource:	6
(while_dsycfvoega_readvariableop_resource: 8
*while_dsycfvoega_readvariableop_1_resource: 8
*while_dsycfvoega_readvariableop_2_resource: ¢'while/dsycfvoega/BiasAdd/ReadVariableOp¢&while/dsycfvoega/MatMul/ReadVariableOp¢(while/dsycfvoega/MatMul_1/ReadVariableOp¢while/dsycfvoega/ReadVariableOp¢!while/dsycfvoega/ReadVariableOp_1¢!while/dsycfvoega/ReadVariableOp_2Ã
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
&while/dsycfvoega/MatMul/ReadVariableOpReadVariableOp1while_dsycfvoega_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dsycfvoega/MatMul/ReadVariableOpÑ
while/dsycfvoega/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMulÉ
(while/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp3while_dsycfvoega_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dsycfvoega/MatMul_1/ReadVariableOpº
while/dsycfvoega/MatMul_1MatMulwhile_placeholder_20while/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMul_1°
while/dsycfvoega/addAddV2!while/dsycfvoega/MatMul:product:0#while/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/addÂ
'while/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp2while_dsycfvoega_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dsycfvoega/BiasAdd/ReadVariableOp½
while/dsycfvoega/BiasAddBiasAddwhile/dsycfvoega/add:z:0/while/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/BiasAdd
 while/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dsycfvoega/split/split_dim
while/dsycfvoega/splitSplit)while/dsycfvoega/split/split_dim:output:0!while/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dsycfvoega/split©
while/dsycfvoega/ReadVariableOpReadVariableOp*while_dsycfvoega_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dsycfvoega/ReadVariableOp£
while/dsycfvoega/mulMul'while/dsycfvoega/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul¦
while/dsycfvoega/add_1AddV2while/dsycfvoega/split:output:0while/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_1
while/dsycfvoega/SigmoidSigmoidwhile/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid¯
!while/dsycfvoega/ReadVariableOp_1ReadVariableOp,while_dsycfvoega_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_1©
while/dsycfvoega/mul_1Mul)while/dsycfvoega/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_1¨
while/dsycfvoega/add_2AddV2while/dsycfvoega/split:output:1while/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_2
while/dsycfvoega/Sigmoid_1Sigmoidwhile/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_1
while/dsycfvoega/mul_2Mulwhile/dsycfvoega/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_2
while/dsycfvoega/TanhTanhwhile/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh¢
while/dsycfvoega/mul_3Mulwhile/dsycfvoega/Sigmoid:y:0while/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_3£
while/dsycfvoega/add_3AddV2while/dsycfvoega/mul_2:z:0while/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_3¯
!while/dsycfvoega/ReadVariableOp_2ReadVariableOp,while_dsycfvoega_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_2°
while/dsycfvoega/mul_4Mul)while/dsycfvoega/ReadVariableOp_2:value:0while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_4¨
while/dsycfvoega/add_4AddV2while/dsycfvoega/split:output:3while/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_4
while/dsycfvoega/Sigmoid_2Sigmoidwhile/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_2
while/dsycfvoega/Tanh_1Tanhwhile/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh_1¦
while/dsycfvoega/mul_5Mulwhile/dsycfvoega/Sigmoid_2:y:0while/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dsycfvoega/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dsycfvoega/mul_5:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dsycfvoega/add_3:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dsycfvoega_biasadd_readvariableop_resource2while_dsycfvoega_biasadd_readvariableop_resource_0"h
1while_dsycfvoega_matmul_1_readvariableop_resource3while_dsycfvoega_matmul_1_readvariableop_resource_0"d
/while_dsycfvoega_matmul_readvariableop_resource1while_dsycfvoega_matmul_readvariableop_resource_0"Z
*while_dsycfvoega_readvariableop_1_resource,while_dsycfvoega_readvariableop_1_resource_0"Z
*while_dsycfvoega_readvariableop_2_resource,while_dsycfvoega_readvariableop_2_resource_0"V
(while_dsycfvoega_readvariableop_resource*while_dsycfvoega_readvariableop_resource_0")
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
'while/dsycfvoega/BiasAdd/ReadVariableOp'while/dsycfvoega/BiasAdd/ReadVariableOp2P
&while/dsycfvoega/MatMul/ReadVariableOp&while/dsycfvoega/MatMul/ReadVariableOp2T
(while/dsycfvoega/MatMul_1/ReadVariableOp(while/dsycfvoega/MatMul_1/ReadVariableOp2B
while/dsycfvoega/ReadVariableOpwhile/dsycfvoega/ReadVariableOp2F
!while/dsycfvoega/ReadVariableOp_1!while/dsycfvoega/ReadVariableOp_12F
!while/dsycfvoega/ReadVariableOp_2!while/dsycfvoega/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_sequential_layer_call_and_return_conditional_losses_657586

inputsL
6bpstkcuudk_conv1d_expanddims_1_readvariableop_resource:K
=bpstkcuudk_squeeze_batch_dims_biasadd_readvariableop_resource:G
4osutmzfngz_dsycfvoega_matmul_readvariableop_resource:	I
6osutmzfngz_dsycfvoega_matmul_1_readvariableop_resource:	 D
5osutmzfngz_dsycfvoega_biasadd_readvariableop_resource:	;
-osutmzfngz_dsycfvoega_readvariableop_resource: =
/osutmzfngz_dsycfvoega_readvariableop_1_resource: =
/osutmzfngz_dsycfvoega_readvariableop_2_resource: G
4owshcilvwl_flzkvrshbq_matmul_readvariableop_resource:	 I
6owshcilvwl_flzkvrshbq_matmul_1_readvariableop_resource:	 D
5owshcilvwl_flzkvrshbq_biasadd_readvariableop_resource:	;
-owshcilvwl_flzkvrshbq_readvariableop_resource: =
/owshcilvwl_flzkvrshbq_readvariableop_1_resource: =
/owshcilvwl_flzkvrshbq_readvariableop_2_resource: ;
)oaettnoaty_matmul_readvariableop_resource: 8
*oaettnoaty_biasadd_readvariableop_resource:
identity¢-bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp¢4bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp¢!oaettnoaty/BiasAdd/ReadVariableOp¢ oaettnoaty/MatMul/ReadVariableOp¢,osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp¢+osutmzfngz/dsycfvoega/MatMul/ReadVariableOp¢-osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp¢$osutmzfngz/dsycfvoega/ReadVariableOp¢&osutmzfngz/dsycfvoega/ReadVariableOp_1¢&osutmzfngz/dsycfvoega/ReadVariableOp_2¢osutmzfngz/while¢,owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp¢+owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp¢-owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp¢$owshcilvwl/flzkvrshbq/ReadVariableOp¢&owshcilvwl/flzkvrshbq/ReadVariableOp_1¢&owshcilvwl/flzkvrshbq/ReadVariableOp_2¢owshcilvwl/while
 bpstkcuudk/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 bpstkcuudk/conv1d/ExpandDims/dim»
bpstkcuudk/conv1d/ExpandDims
ExpandDimsinputs)bpstkcuudk/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
bpstkcuudk/conv1d/ExpandDimsÙ
-bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6bpstkcuudk_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp
"bpstkcuudk/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"bpstkcuudk/conv1d/ExpandDims_1/dimã
bpstkcuudk/conv1d/ExpandDims_1
ExpandDims5bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp:value:0+bpstkcuudk/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
bpstkcuudk/conv1d/ExpandDims_1
bpstkcuudk/conv1d/ShapeShape%bpstkcuudk/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
bpstkcuudk/conv1d/Shape
%bpstkcuudk/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%bpstkcuudk/conv1d/strided_slice/stack¥
'bpstkcuudk/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'bpstkcuudk/conv1d/strided_slice/stack_1
'bpstkcuudk/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'bpstkcuudk/conv1d/strided_slice/stack_2Ì
bpstkcuudk/conv1d/strided_sliceStridedSlice bpstkcuudk/conv1d/Shape:output:0.bpstkcuudk/conv1d/strided_slice/stack:output:00bpstkcuudk/conv1d/strided_slice/stack_1:output:00bpstkcuudk/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
bpstkcuudk/conv1d/strided_slice
bpstkcuudk/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
bpstkcuudk/conv1d/Reshape/shapeÌ
bpstkcuudk/conv1d/ReshapeReshape%bpstkcuudk/conv1d/ExpandDims:output:0(bpstkcuudk/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bpstkcuudk/conv1d/Reshapeî
bpstkcuudk/conv1d/Conv2DConv2D"bpstkcuudk/conv1d/Reshape:output:0'bpstkcuudk/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
bpstkcuudk/conv1d/Conv2D
!bpstkcuudk/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!bpstkcuudk/conv1d/concat/values_1
bpstkcuudk/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
bpstkcuudk/conv1d/concat/axisì
bpstkcuudk/conv1d/concatConcatV2(bpstkcuudk/conv1d/strided_slice:output:0*bpstkcuudk/conv1d/concat/values_1:output:0&bpstkcuudk/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
bpstkcuudk/conv1d/concatÉ
bpstkcuudk/conv1d/Reshape_1Reshape!bpstkcuudk/conv1d/Conv2D:output:0!bpstkcuudk/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
bpstkcuudk/conv1d/Reshape_1Á
bpstkcuudk/conv1d/SqueezeSqueeze$bpstkcuudk/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
bpstkcuudk/conv1d/Squeeze
#bpstkcuudk/squeeze_batch_dims/ShapeShape"bpstkcuudk/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#bpstkcuudk/squeeze_batch_dims/Shape°
1bpstkcuudk/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1bpstkcuudk/squeeze_batch_dims/strided_slice/stack½
3bpstkcuudk/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3bpstkcuudk/squeeze_batch_dims/strided_slice/stack_1´
3bpstkcuudk/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3bpstkcuudk/squeeze_batch_dims/strided_slice/stack_2
+bpstkcuudk/squeeze_batch_dims/strided_sliceStridedSlice,bpstkcuudk/squeeze_batch_dims/Shape:output:0:bpstkcuudk/squeeze_batch_dims/strided_slice/stack:output:0<bpstkcuudk/squeeze_batch_dims/strided_slice/stack_1:output:0<bpstkcuudk/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+bpstkcuudk/squeeze_batch_dims/strided_slice¯
+bpstkcuudk/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+bpstkcuudk/squeeze_batch_dims/Reshape/shapeé
%bpstkcuudk/squeeze_batch_dims/ReshapeReshape"bpstkcuudk/conv1d/Squeeze:output:04bpstkcuudk/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%bpstkcuudk/squeeze_batch_dims/Reshapeæ
4bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=bpstkcuudk_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%bpstkcuudk/squeeze_batch_dims/BiasAddBiasAdd.bpstkcuudk/squeeze_batch_dims/Reshape:output:0<bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%bpstkcuudk/squeeze_batch_dims/BiasAdd¯
-bpstkcuudk/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-bpstkcuudk/squeeze_batch_dims/concat/values_1¡
)bpstkcuudk/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)bpstkcuudk/squeeze_batch_dims/concat/axis¨
$bpstkcuudk/squeeze_batch_dims/concatConcatV24bpstkcuudk/squeeze_batch_dims/strided_slice:output:06bpstkcuudk/squeeze_batch_dims/concat/values_1:output:02bpstkcuudk/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$bpstkcuudk/squeeze_batch_dims/concatö
'bpstkcuudk/squeeze_batch_dims/Reshape_1Reshape.bpstkcuudk/squeeze_batch_dims/BiasAdd:output:0-bpstkcuudk/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'bpstkcuudk/squeeze_batch_dims/Reshape_1
xlcvyoxoxq/ShapeShape0bpstkcuudk/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
xlcvyoxoxq/Shape
xlcvyoxoxq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
xlcvyoxoxq/strided_slice/stack
 xlcvyoxoxq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 xlcvyoxoxq/strided_slice/stack_1
 xlcvyoxoxq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 xlcvyoxoxq/strided_slice/stack_2¤
xlcvyoxoxq/strided_sliceStridedSlicexlcvyoxoxq/Shape:output:0'xlcvyoxoxq/strided_slice/stack:output:0)xlcvyoxoxq/strided_slice/stack_1:output:0)xlcvyoxoxq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
xlcvyoxoxq/strided_slicez
xlcvyoxoxq/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
xlcvyoxoxq/Reshape/shape/1z
xlcvyoxoxq/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
xlcvyoxoxq/Reshape/shape/2×
xlcvyoxoxq/Reshape/shapePack!xlcvyoxoxq/strided_slice:output:0#xlcvyoxoxq/Reshape/shape/1:output:0#xlcvyoxoxq/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
xlcvyoxoxq/Reshape/shape¾
xlcvyoxoxq/ReshapeReshape0bpstkcuudk/squeeze_batch_dims/Reshape_1:output:0!xlcvyoxoxq/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xlcvyoxoxq/Reshapeo
osutmzfngz/ShapeShapexlcvyoxoxq/Reshape:output:0*
T0*
_output_shapes
:2
osutmzfngz/Shape
osutmzfngz/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
osutmzfngz/strided_slice/stack
 osutmzfngz/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 osutmzfngz/strided_slice/stack_1
 osutmzfngz/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 osutmzfngz/strided_slice/stack_2¤
osutmzfngz/strided_sliceStridedSliceosutmzfngz/Shape:output:0'osutmzfngz/strided_slice/stack:output:0)osutmzfngz/strided_slice/stack_1:output:0)osutmzfngz/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
osutmzfngz/strided_slicer
osutmzfngz/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/zeros/mul/y
osutmzfngz/zeros/mulMul!osutmzfngz/strided_slice:output:0osutmzfngz/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/zeros/mulu
osutmzfngz/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
osutmzfngz/zeros/Less/y
osutmzfngz/zeros/LessLessosutmzfngz/zeros/mul:z:0 osutmzfngz/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/zeros/Lessx
osutmzfngz/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/zeros/packed/1¯
osutmzfngz/zeros/packedPack!osutmzfngz/strided_slice:output:0"osutmzfngz/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
osutmzfngz/zeros/packedu
osutmzfngz/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
osutmzfngz/zeros/Const¡
osutmzfngz/zerosFill osutmzfngz/zeros/packed:output:0osutmzfngz/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/zerosv
osutmzfngz/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/zeros_1/mul/y
osutmzfngz/zeros_1/mulMul!osutmzfngz/strided_slice:output:0!osutmzfngz/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/zeros_1/muly
osutmzfngz/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
osutmzfngz/zeros_1/Less/y
osutmzfngz/zeros_1/LessLessosutmzfngz/zeros_1/mul:z:0"osutmzfngz/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
osutmzfngz/zeros_1/Less|
osutmzfngz/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/zeros_1/packed/1µ
osutmzfngz/zeros_1/packedPack!osutmzfngz/strided_slice:output:0$osutmzfngz/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
osutmzfngz/zeros_1/packedy
osutmzfngz/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
osutmzfngz/zeros_1/Const©
osutmzfngz/zeros_1Fill"osutmzfngz/zeros_1/packed:output:0!osutmzfngz/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/zeros_1
osutmzfngz/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
osutmzfngz/transpose/perm°
osutmzfngz/transpose	Transposexlcvyoxoxq/Reshape:output:0"osutmzfngz/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
osutmzfngz/transposep
osutmzfngz/Shape_1Shapeosutmzfngz/transpose:y:0*
T0*
_output_shapes
:2
osutmzfngz/Shape_1
 osutmzfngz/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 osutmzfngz/strided_slice_1/stack
"osutmzfngz/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"osutmzfngz/strided_slice_1/stack_1
"osutmzfngz/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"osutmzfngz/strided_slice_1/stack_2°
osutmzfngz/strided_slice_1StridedSliceosutmzfngz/Shape_1:output:0)osutmzfngz/strided_slice_1/stack:output:0+osutmzfngz/strided_slice_1/stack_1:output:0+osutmzfngz/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
osutmzfngz/strided_slice_1
&osutmzfngz/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&osutmzfngz/TensorArrayV2/element_shapeÞ
osutmzfngz/TensorArrayV2TensorListReserve/osutmzfngz/TensorArrayV2/element_shape:output:0#osutmzfngz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
osutmzfngz/TensorArrayV2Õ
@osutmzfngz/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@osutmzfngz/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2osutmzfngz/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorosutmzfngz/transpose:y:0Iosutmzfngz/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2osutmzfngz/TensorArrayUnstack/TensorListFromTensor
 osutmzfngz/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 osutmzfngz/strided_slice_2/stack
"osutmzfngz/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"osutmzfngz/strided_slice_2/stack_1
"osutmzfngz/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"osutmzfngz/strided_slice_2/stack_2¾
osutmzfngz/strided_slice_2StridedSliceosutmzfngz/transpose:y:0)osutmzfngz/strided_slice_2/stack:output:0+osutmzfngz/strided_slice_2/stack_1:output:0+osutmzfngz/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
osutmzfngz/strided_slice_2Ð
+osutmzfngz/dsycfvoega/MatMul/ReadVariableOpReadVariableOp4osutmzfngz_dsycfvoega_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+osutmzfngz/dsycfvoega/MatMul/ReadVariableOpÓ
osutmzfngz/dsycfvoega/MatMulMatMul#osutmzfngz/strided_slice_2:output:03osutmzfngz/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
osutmzfngz/dsycfvoega/MatMulÖ
-osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp6osutmzfngz_dsycfvoega_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOpÏ
osutmzfngz/dsycfvoega/MatMul_1MatMulosutmzfngz/zeros:output:05osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
osutmzfngz/dsycfvoega/MatMul_1Ä
osutmzfngz/dsycfvoega/addAddV2&osutmzfngz/dsycfvoega/MatMul:product:0(osutmzfngz/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
osutmzfngz/dsycfvoega/addÏ
,osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp5osutmzfngz_dsycfvoega_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOpÑ
osutmzfngz/dsycfvoega/BiasAddBiasAddosutmzfngz/dsycfvoega/add:z:04osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
osutmzfngz/dsycfvoega/BiasAdd
%osutmzfngz/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%osutmzfngz/dsycfvoega/split/split_dim
osutmzfngz/dsycfvoega/splitSplit.osutmzfngz/dsycfvoega/split/split_dim:output:0&osutmzfngz/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
osutmzfngz/dsycfvoega/split¶
$osutmzfngz/dsycfvoega/ReadVariableOpReadVariableOp-osutmzfngz_dsycfvoega_readvariableop_resource*
_output_shapes
: *
dtype02&
$osutmzfngz/dsycfvoega/ReadVariableOpº
osutmzfngz/dsycfvoega/mulMul,osutmzfngz/dsycfvoega/ReadVariableOp:value:0osutmzfngz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mulº
osutmzfngz/dsycfvoega/add_1AddV2$osutmzfngz/dsycfvoega/split:output:0osutmzfngz/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/add_1
osutmzfngz/dsycfvoega/SigmoidSigmoidosutmzfngz/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/Sigmoid¼
&osutmzfngz/dsycfvoega/ReadVariableOp_1ReadVariableOp/osutmzfngz_dsycfvoega_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&osutmzfngz/dsycfvoega/ReadVariableOp_1À
osutmzfngz/dsycfvoega/mul_1Mul.osutmzfngz/dsycfvoega/ReadVariableOp_1:value:0osutmzfngz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mul_1¼
osutmzfngz/dsycfvoega/add_2AddV2$osutmzfngz/dsycfvoega/split:output:1osutmzfngz/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/add_2 
osutmzfngz/dsycfvoega/Sigmoid_1Sigmoidosutmzfngz/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
osutmzfngz/dsycfvoega/Sigmoid_1µ
osutmzfngz/dsycfvoega/mul_2Mul#osutmzfngz/dsycfvoega/Sigmoid_1:y:0osutmzfngz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mul_2
osutmzfngz/dsycfvoega/TanhTanh$osutmzfngz/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/Tanh¶
osutmzfngz/dsycfvoega/mul_3Mul!osutmzfngz/dsycfvoega/Sigmoid:y:0osutmzfngz/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mul_3·
osutmzfngz/dsycfvoega/add_3AddV2osutmzfngz/dsycfvoega/mul_2:z:0osutmzfngz/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/add_3¼
&osutmzfngz/dsycfvoega/ReadVariableOp_2ReadVariableOp/osutmzfngz_dsycfvoega_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&osutmzfngz/dsycfvoega/ReadVariableOp_2Ä
osutmzfngz/dsycfvoega/mul_4Mul.osutmzfngz/dsycfvoega/ReadVariableOp_2:value:0osutmzfngz/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mul_4¼
osutmzfngz/dsycfvoega/add_4AddV2$osutmzfngz/dsycfvoega/split:output:3osutmzfngz/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/add_4 
osutmzfngz/dsycfvoega/Sigmoid_2Sigmoidosutmzfngz/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
osutmzfngz/dsycfvoega/Sigmoid_2
osutmzfngz/dsycfvoega/Tanh_1Tanhosutmzfngz/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/Tanh_1º
osutmzfngz/dsycfvoega/mul_5Mul#osutmzfngz/dsycfvoega/Sigmoid_2:y:0 osutmzfngz/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/dsycfvoega/mul_5¥
(osutmzfngz/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(osutmzfngz/TensorArrayV2_1/element_shapeä
osutmzfngz/TensorArrayV2_1TensorListReserve1osutmzfngz/TensorArrayV2_1/element_shape:output:0#osutmzfngz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
osutmzfngz/TensorArrayV2_1d
osutmzfngz/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/time
#osutmzfngz/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#osutmzfngz/while/maximum_iterations
osutmzfngz/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
osutmzfngz/while/loop_counter°
osutmzfngz/whileWhile&osutmzfngz/while/loop_counter:output:0,osutmzfngz/while/maximum_iterations:output:0osutmzfngz/time:output:0#osutmzfngz/TensorArrayV2_1:handle:0osutmzfngz/zeros:output:0osutmzfngz/zeros_1:output:0#osutmzfngz/strided_slice_1:output:0Bosutmzfngz/TensorArrayUnstack/TensorListFromTensor:output_handle:04osutmzfngz_dsycfvoega_matmul_readvariableop_resource6osutmzfngz_dsycfvoega_matmul_1_readvariableop_resource5osutmzfngz_dsycfvoega_biasadd_readvariableop_resource-osutmzfngz_dsycfvoega_readvariableop_resource/osutmzfngz_dsycfvoega_readvariableop_1_resource/osutmzfngz_dsycfvoega_readvariableop_2_resource*
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
osutmzfngz_while_body_657303*(
cond R
osutmzfngz_while_cond_657302*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
osutmzfngz/whileË
;osutmzfngz/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;osutmzfngz/TensorArrayV2Stack/TensorListStack/element_shape
-osutmzfngz/TensorArrayV2Stack/TensorListStackTensorListStackosutmzfngz/while:output:3Dosutmzfngz/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-osutmzfngz/TensorArrayV2Stack/TensorListStack
 osutmzfngz/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 osutmzfngz/strided_slice_3/stack
"osutmzfngz/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"osutmzfngz/strided_slice_3/stack_1
"osutmzfngz/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"osutmzfngz/strided_slice_3/stack_2Ü
osutmzfngz/strided_slice_3StridedSlice6osutmzfngz/TensorArrayV2Stack/TensorListStack:tensor:0)osutmzfngz/strided_slice_3/stack:output:0+osutmzfngz/strided_slice_3/stack_1:output:0+osutmzfngz/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
osutmzfngz/strided_slice_3
osutmzfngz/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
osutmzfngz/transpose_1/permÑ
osutmzfngz/transpose_1	Transpose6osutmzfngz/TensorArrayV2Stack/TensorListStack:tensor:0$osutmzfngz/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
osutmzfngz/transpose_1n
owshcilvwl/ShapeShapeosutmzfngz/transpose_1:y:0*
T0*
_output_shapes
:2
owshcilvwl/Shape
owshcilvwl/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
owshcilvwl/strided_slice/stack
 owshcilvwl/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 owshcilvwl/strided_slice/stack_1
 owshcilvwl/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 owshcilvwl/strided_slice/stack_2¤
owshcilvwl/strided_sliceStridedSliceowshcilvwl/Shape:output:0'owshcilvwl/strided_slice/stack:output:0)owshcilvwl/strided_slice/stack_1:output:0)owshcilvwl/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
owshcilvwl/strided_slicer
owshcilvwl/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/zeros/mul/y
owshcilvwl/zeros/mulMul!owshcilvwl/strided_slice:output:0owshcilvwl/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/zeros/mulu
owshcilvwl/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
owshcilvwl/zeros/Less/y
owshcilvwl/zeros/LessLessowshcilvwl/zeros/mul:z:0 owshcilvwl/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/zeros/Lessx
owshcilvwl/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/zeros/packed/1¯
owshcilvwl/zeros/packedPack!owshcilvwl/strided_slice:output:0"owshcilvwl/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
owshcilvwl/zeros/packedu
owshcilvwl/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
owshcilvwl/zeros/Const¡
owshcilvwl/zerosFill owshcilvwl/zeros/packed:output:0owshcilvwl/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/zerosv
owshcilvwl/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/zeros_1/mul/y
owshcilvwl/zeros_1/mulMul!owshcilvwl/strided_slice:output:0!owshcilvwl/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/zeros_1/muly
owshcilvwl/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
owshcilvwl/zeros_1/Less/y
owshcilvwl/zeros_1/LessLessowshcilvwl/zeros_1/mul:z:0"owshcilvwl/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
owshcilvwl/zeros_1/Less|
owshcilvwl/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/zeros_1/packed/1µ
owshcilvwl/zeros_1/packedPack!owshcilvwl/strided_slice:output:0$owshcilvwl/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
owshcilvwl/zeros_1/packedy
owshcilvwl/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
owshcilvwl/zeros_1/Const©
owshcilvwl/zeros_1Fill"owshcilvwl/zeros_1/packed:output:0!owshcilvwl/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/zeros_1
owshcilvwl/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
owshcilvwl/transpose/perm¯
owshcilvwl/transpose	Transposeosutmzfngz/transpose_1:y:0"owshcilvwl/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/transposep
owshcilvwl/Shape_1Shapeowshcilvwl/transpose:y:0*
T0*
_output_shapes
:2
owshcilvwl/Shape_1
 owshcilvwl/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 owshcilvwl/strided_slice_1/stack
"owshcilvwl/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"owshcilvwl/strided_slice_1/stack_1
"owshcilvwl/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"owshcilvwl/strided_slice_1/stack_2°
owshcilvwl/strided_slice_1StridedSliceowshcilvwl/Shape_1:output:0)owshcilvwl/strided_slice_1/stack:output:0+owshcilvwl/strided_slice_1/stack_1:output:0+owshcilvwl/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
owshcilvwl/strided_slice_1
&owshcilvwl/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&owshcilvwl/TensorArrayV2/element_shapeÞ
owshcilvwl/TensorArrayV2TensorListReserve/owshcilvwl/TensorArrayV2/element_shape:output:0#owshcilvwl/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
owshcilvwl/TensorArrayV2Õ
@owshcilvwl/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@owshcilvwl/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2owshcilvwl/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorowshcilvwl/transpose:y:0Iowshcilvwl/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2owshcilvwl/TensorArrayUnstack/TensorListFromTensor
 owshcilvwl/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 owshcilvwl/strided_slice_2/stack
"owshcilvwl/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"owshcilvwl/strided_slice_2/stack_1
"owshcilvwl/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"owshcilvwl/strided_slice_2/stack_2¾
owshcilvwl/strided_slice_2StridedSliceowshcilvwl/transpose:y:0)owshcilvwl/strided_slice_2/stack:output:0+owshcilvwl/strided_slice_2/stack_1:output:0+owshcilvwl/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
owshcilvwl/strided_slice_2Ð
+owshcilvwl/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp4owshcilvwl_flzkvrshbq_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+owshcilvwl/flzkvrshbq/MatMul/ReadVariableOpÓ
owshcilvwl/flzkvrshbq/MatMulMatMul#owshcilvwl/strided_slice_2:output:03owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
owshcilvwl/flzkvrshbq/MatMulÖ
-owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp6owshcilvwl_flzkvrshbq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOpÏ
owshcilvwl/flzkvrshbq/MatMul_1MatMulowshcilvwl/zeros:output:05owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
owshcilvwl/flzkvrshbq/MatMul_1Ä
owshcilvwl/flzkvrshbq/addAddV2&owshcilvwl/flzkvrshbq/MatMul:product:0(owshcilvwl/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
owshcilvwl/flzkvrshbq/addÏ
,owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp5owshcilvwl_flzkvrshbq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOpÑ
owshcilvwl/flzkvrshbq/BiasAddBiasAddowshcilvwl/flzkvrshbq/add:z:04owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
owshcilvwl/flzkvrshbq/BiasAdd
%owshcilvwl/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%owshcilvwl/flzkvrshbq/split/split_dim
owshcilvwl/flzkvrshbq/splitSplit.owshcilvwl/flzkvrshbq/split/split_dim:output:0&owshcilvwl/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
owshcilvwl/flzkvrshbq/split¶
$owshcilvwl/flzkvrshbq/ReadVariableOpReadVariableOp-owshcilvwl_flzkvrshbq_readvariableop_resource*
_output_shapes
: *
dtype02&
$owshcilvwl/flzkvrshbq/ReadVariableOpº
owshcilvwl/flzkvrshbq/mulMul,owshcilvwl/flzkvrshbq/ReadVariableOp:value:0owshcilvwl/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mulº
owshcilvwl/flzkvrshbq/add_1AddV2$owshcilvwl/flzkvrshbq/split:output:0owshcilvwl/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/add_1
owshcilvwl/flzkvrshbq/SigmoidSigmoidowshcilvwl/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/Sigmoid¼
&owshcilvwl/flzkvrshbq/ReadVariableOp_1ReadVariableOp/owshcilvwl_flzkvrshbq_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&owshcilvwl/flzkvrshbq/ReadVariableOp_1À
owshcilvwl/flzkvrshbq/mul_1Mul.owshcilvwl/flzkvrshbq/ReadVariableOp_1:value:0owshcilvwl/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mul_1¼
owshcilvwl/flzkvrshbq/add_2AddV2$owshcilvwl/flzkvrshbq/split:output:1owshcilvwl/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/add_2 
owshcilvwl/flzkvrshbq/Sigmoid_1Sigmoidowshcilvwl/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
owshcilvwl/flzkvrshbq/Sigmoid_1µ
owshcilvwl/flzkvrshbq/mul_2Mul#owshcilvwl/flzkvrshbq/Sigmoid_1:y:0owshcilvwl/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mul_2
owshcilvwl/flzkvrshbq/TanhTanh$owshcilvwl/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/Tanh¶
owshcilvwl/flzkvrshbq/mul_3Mul!owshcilvwl/flzkvrshbq/Sigmoid:y:0owshcilvwl/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mul_3·
owshcilvwl/flzkvrshbq/add_3AddV2owshcilvwl/flzkvrshbq/mul_2:z:0owshcilvwl/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/add_3¼
&owshcilvwl/flzkvrshbq/ReadVariableOp_2ReadVariableOp/owshcilvwl_flzkvrshbq_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&owshcilvwl/flzkvrshbq/ReadVariableOp_2Ä
owshcilvwl/flzkvrshbq/mul_4Mul.owshcilvwl/flzkvrshbq/ReadVariableOp_2:value:0owshcilvwl/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mul_4¼
owshcilvwl/flzkvrshbq/add_4AddV2$owshcilvwl/flzkvrshbq/split:output:3owshcilvwl/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/add_4 
owshcilvwl/flzkvrshbq/Sigmoid_2Sigmoidowshcilvwl/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
owshcilvwl/flzkvrshbq/Sigmoid_2
owshcilvwl/flzkvrshbq/Tanh_1Tanhowshcilvwl/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/Tanh_1º
owshcilvwl/flzkvrshbq/mul_5Mul#owshcilvwl/flzkvrshbq/Sigmoid_2:y:0 owshcilvwl/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/flzkvrshbq/mul_5¥
(owshcilvwl/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(owshcilvwl/TensorArrayV2_1/element_shapeä
owshcilvwl/TensorArrayV2_1TensorListReserve1owshcilvwl/TensorArrayV2_1/element_shape:output:0#owshcilvwl/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
owshcilvwl/TensorArrayV2_1d
owshcilvwl/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/time
#owshcilvwl/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#owshcilvwl/while/maximum_iterations
owshcilvwl/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
owshcilvwl/while/loop_counter°
owshcilvwl/whileWhile&owshcilvwl/while/loop_counter:output:0,owshcilvwl/while/maximum_iterations:output:0owshcilvwl/time:output:0#owshcilvwl/TensorArrayV2_1:handle:0owshcilvwl/zeros:output:0owshcilvwl/zeros_1:output:0#owshcilvwl/strided_slice_1:output:0Bowshcilvwl/TensorArrayUnstack/TensorListFromTensor:output_handle:04owshcilvwl_flzkvrshbq_matmul_readvariableop_resource6owshcilvwl_flzkvrshbq_matmul_1_readvariableop_resource5owshcilvwl_flzkvrshbq_biasadd_readvariableop_resource-owshcilvwl_flzkvrshbq_readvariableop_resource/owshcilvwl_flzkvrshbq_readvariableop_1_resource/owshcilvwl_flzkvrshbq_readvariableop_2_resource*
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
owshcilvwl_while_body_657479*(
cond R
owshcilvwl_while_cond_657478*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
owshcilvwl/whileË
;owshcilvwl/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;owshcilvwl/TensorArrayV2Stack/TensorListStack/element_shape
-owshcilvwl/TensorArrayV2Stack/TensorListStackTensorListStackowshcilvwl/while:output:3Dowshcilvwl/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-owshcilvwl/TensorArrayV2Stack/TensorListStack
 owshcilvwl/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 owshcilvwl/strided_slice_3/stack
"owshcilvwl/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"owshcilvwl/strided_slice_3/stack_1
"owshcilvwl/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"owshcilvwl/strided_slice_3/stack_2Ü
owshcilvwl/strided_slice_3StridedSlice6owshcilvwl/TensorArrayV2Stack/TensorListStack:tensor:0)owshcilvwl/strided_slice_3/stack:output:0+owshcilvwl/strided_slice_3/stack_1:output:0+owshcilvwl/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
owshcilvwl/strided_slice_3
owshcilvwl/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
owshcilvwl/transpose_1/permÑ
owshcilvwl/transpose_1	Transpose6owshcilvwl/TensorArrayV2Stack/TensorListStack:tensor:0$owshcilvwl/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
owshcilvwl/transpose_1®
 oaettnoaty/MatMul/ReadVariableOpReadVariableOp)oaettnoaty_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 oaettnoaty/MatMul/ReadVariableOp±
oaettnoaty/MatMulMatMul#owshcilvwl/strided_slice_3:output:0(oaettnoaty/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oaettnoaty/MatMul­
!oaettnoaty/BiasAdd/ReadVariableOpReadVariableOp*oaettnoaty_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!oaettnoaty/BiasAdd/ReadVariableOp­
oaettnoaty/BiasAddBiasAddoaettnoaty/MatMul:product:0)oaettnoaty/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oaettnoaty/BiasAddÏ
IdentityIdentityoaettnoaty/BiasAdd:output:0.^bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp5^bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp"^oaettnoaty/BiasAdd/ReadVariableOp!^oaettnoaty/MatMul/ReadVariableOp-^osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp,^osutmzfngz/dsycfvoega/MatMul/ReadVariableOp.^osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp%^osutmzfngz/dsycfvoega/ReadVariableOp'^osutmzfngz/dsycfvoega/ReadVariableOp_1'^osutmzfngz/dsycfvoega/ReadVariableOp_2^osutmzfngz/while-^owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp,^owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp.^owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp%^owshcilvwl/flzkvrshbq/ReadVariableOp'^owshcilvwl/flzkvrshbq/ReadVariableOp_1'^owshcilvwl/flzkvrshbq/ReadVariableOp_2^owshcilvwl/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2^
-bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp-bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp2l
4bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp4bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!oaettnoaty/BiasAdd/ReadVariableOp!oaettnoaty/BiasAdd/ReadVariableOp2D
 oaettnoaty/MatMul/ReadVariableOp oaettnoaty/MatMul/ReadVariableOp2\
,osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp,osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp2Z
+osutmzfngz/dsycfvoega/MatMul/ReadVariableOp+osutmzfngz/dsycfvoega/MatMul/ReadVariableOp2^
-osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp-osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp2L
$osutmzfngz/dsycfvoega/ReadVariableOp$osutmzfngz/dsycfvoega/ReadVariableOp2P
&osutmzfngz/dsycfvoega/ReadVariableOp_1&osutmzfngz/dsycfvoega/ReadVariableOp_12P
&osutmzfngz/dsycfvoega/ReadVariableOp_2&osutmzfngz/dsycfvoega/ReadVariableOp_22$
osutmzfngz/whileosutmzfngz/while2\
,owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp,owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp2Z
+owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp+owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp2^
-owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp-owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp2L
$owshcilvwl/flzkvrshbq/ReadVariableOp$owshcilvwl/flzkvrshbq/ReadVariableOp2P
&owshcilvwl/flzkvrshbq/ReadVariableOp_1&owshcilvwl/flzkvrshbq/ReadVariableOp_12P
&owshcilvwl/flzkvrshbq/ReadVariableOp_2&owshcilvwl/flzkvrshbq/ReadVariableOp_22$
owshcilvwl/whileowshcilvwl/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

«
F__inference_sequential_layer_call_and_return_conditional_losses_656505

inputs'
bpstkcuudk_656467:
bpstkcuudk_656469:$
osutmzfngz_656473:	$
osutmzfngz_656475:	  
osutmzfngz_656477:	
osutmzfngz_656479: 
osutmzfngz_656481: 
osutmzfngz_656483: $
owshcilvwl_656486:	 $
owshcilvwl_656488:	  
owshcilvwl_656490:	
owshcilvwl_656492: 
owshcilvwl_656494: 
owshcilvwl_656496: #
oaettnoaty_656499: 
oaettnoaty_656501:
identity¢"bpstkcuudk/StatefulPartitionedCall¢"oaettnoaty/StatefulPartitionedCall¢"osutmzfngz/StatefulPartitionedCall¢"owshcilvwl/StatefulPartitionedCall©
"bpstkcuudk/StatefulPartitionedCallStatefulPartitionedCallinputsbpstkcuudk_656467bpstkcuudk_656469*
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
F__inference_bpstkcuudk_layer_call_and_return_conditional_losses_6555122$
"bpstkcuudk/StatefulPartitionedCall
xlcvyoxoxq/PartitionedCallPartitionedCall+bpstkcuudk/StatefulPartitionedCall:output:0*
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
F__inference_xlcvyoxoxq_layer_call_and_return_conditional_losses_6555312
xlcvyoxoxq/PartitionedCall
"osutmzfngz/StatefulPartitionedCallStatefulPartitionedCall#xlcvyoxoxq/PartitionedCall:output:0osutmzfngz_656473osutmzfngz_656475osutmzfngz_656477osutmzfngz_656479osutmzfngz_656481osutmzfngz_656483*
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_6563942$
"osutmzfngz/StatefulPartitionedCall
"owshcilvwl/StatefulPartitionedCallStatefulPartitionedCall+osutmzfngz/StatefulPartitionedCall:output:0owshcilvwl_656486owshcilvwl_656488owshcilvwl_656490owshcilvwl_656492owshcilvwl_656494owshcilvwl_656496*
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_6561802$
"owshcilvwl/StatefulPartitionedCallÆ
"oaettnoaty/StatefulPartitionedCallStatefulPartitionedCall+owshcilvwl/StatefulPartitionedCall:output:0oaettnoaty_656499oaettnoaty_656501*
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
F__inference_oaettnoaty_layer_call_and_return_conditional_losses_6559292$
"oaettnoaty/StatefulPartitionedCall
IdentityIdentity+oaettnoaty/StatefulPartitionedCall:output:0#^bpstkcuudk/StatefulPartitionedCall#^oaettnoaty/StatefulPartitionedCall#^osutmzfngz/StatefulPartitionedCall#^owshcilvwl/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"bpstkcuudk/StatefulPartitionedCall"bpstkcuudk/StatefulPartitionedCall2H
"oaettnoaty/StatefulPartitionedCall"oaettnoaty/StatefulPartitionedCall2H
"osutmzfngz/StatefulPartitionedCall"osutmzfngz/StatefulPartitionedCall2H
"owshcilvwl/StatefulPartitionedCall"owshcilvwl/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç)
Å
while_body_654322
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_dsycfvoega_654346_0:	,
while_dsycfvoega_654348_0:	 (
while_dsycfvoega_654350_0:	'
while_dsycfvoega_654352_0: '
while_dsycfvoega_654354_0: '
while_dsycfvoega_654356_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_dsycfvoega_654346:	*
while_dsycfvoega_654348:	 &
while_dsycfvoega_654350:	%
while_dsycfvoega_654352: %
while_dsycfvoega_654354: %
while_dsycfvoega_654356: ¢(while/dsycfvoega/StatefulPartitionedCallÃ
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
(while/dsycfvoega/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_dsycfvoega_654346_0while_dsycfvoega_654348_0while_dsycfvoega_654350_0while_dsycfvoega_654352_0while_dsycfvoega_654354_0while_dsycfvoega_654356_0*
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
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_6542262*
(while/dsycfvoega/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/dsycfvoega/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/dsycfvoega/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/dsycfvoega/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/dsycfvoega/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/dsycfvoega/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/dsycfvoega/StatefulPartitionedCall:output:1)^while/dsycfvoega/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/dsycfvoega/StatefulPartitionedCall:output:2)^while/dsycfvoega/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_dsycfvoega_654346while_dsycfvoega_654346_0"4
while_dsycfvoega_654348while_dsycfvoega_654348_0"4
while_dsycfvoega_654350while_dsycfvoega_654350_0"4
while_dsycfvoega_654352while_dsycfvoega_654352_0"4
while_dsycfvoega_654354while_dsycfvoega_654354_0"4
while_dsycfvoega_654356while_dsycfvoega_654356_0")
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
(while/dsycfvoega/StatefulPartitionedCall(while/dsycfvoega/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
osutmzfngz_while_cond_6573022
.osutmzfngz_while_osutmzfngz_while_loop_counter8
4osutmzfngz_while_osutmzfngz_while_maximum_iterations 
osutmzfngz_while_placeholder"
osutmzfngz_while_placeholder_1"
osutmzfngz_while_placeholder_2"
osutmzfngz_while_placeholder_34
0osutmzfngz_while_less_osutmzfngz_strided_slice_1J
Fosutmzfngz_while_osutmzfngz_while_cond_657302___redundant_placeholder0J
Fosutmzfngz_while_osutmzfngz_while_cond_657302___redundant_placeholder1J
Fosutmzfngz_while_osutmzfngz_while_cond_657302___redundant_placeholder2J
Fosutmzfngz_while_osutmzfngz_while_cond_657302___redundant_placeholder3J
Fosutmzfngz_while_osutmzfngz_while_cond_657302___redundant_placeholder4J
Fosutmzfngz_while_osutmzfngz_while_cond_657302___redundant_placeholder5J
Fosutmzfngz_while_osutmzfngz_while_cond_657302___redundant_placeholder6
osutmzfngz_while_identity
§
osutmzfngz/while/LessLessosutmzfngz_while_placeholder0osutmzfngz_while_less_osutmzfngz_strided_slice_1*
T0*
_output_shapes
: 2
osutmzfngz/while/Less~
osutmzfngz/while/IdentityIdentityosutmzfngz/while/Less:z:0*
T0
*
_output_shapes
: 2
osutmzfngz/while/Identity"?
osutmzfngz_while_identity"osutmzfngz/while/Identity:output:0*(
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
F__inference_sequential_layer_call_and_return_conditional_losses_655936

inputs'
bpstkcuudk_655513:
bpstkcuudk_655515:$
osutmzfngz_655713:	$
osutmzfngz_655715:	  
osutmzfngz_655717:	
osutmzfngz_655719: 
osutmzfngz_655721: 
osutmzfngz_655723: $
owshcilvwl_655906:	 $
owshcilvwl_655908:	  
owshcilvwl_655910:	
owshcilvwl_655912: 
owshcilvwl_655914: 
owshcilvwl_655916: #
oaettnoaty_655930: 
oaettnoaty_655932:
identity¢"bpstkcuudk/StatefulPartitionedCall¢"oaettnoaty/StatefulPartitionedCall¢"osutmzfngz/StatefulPartitionedCall¢"owshcilvwl/StatefulPartitionedCall©
"bpstkcuudk/StatefulPartitionedCallStatefulPartitionedCallinputsbpstkcuudk_655513bpstkcuudk_655515*
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
F__inference_bpstkcuudk_layer_call_and_return_conditional_losses_6555122$
"bpstkcuudk/StatefulPartitionedCall
xlcvyoxoxq/PartitionedCallPartitionedCall+bpstkcuudk/StatefulPartitionedCall:output:0*
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
F__inference_xlcvyoxoxq_layer_call_and_return_conditional_losses_6555312
xlcvyoxoxq/PartitionedCall
"osutmzfngz/StatefulPartitionedCallStatefulPartitionedCall#xlcvyoxoxq/PartitionedCall:output:0osutmzfngz_655713osutmzfngz_655715osutmzfngz_655717osutmzfngz_655719osutmzfngz_655721osutmzfngz_655723*
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_6557122$
"osutmzfngz/StatefulPartitionedCall
"owshcilvwl/StatefulPartitionedCallStatefulPartitionedCall+osutmzfngz/StatefulPartitionedCall:output:0owshcilvwl_655906owshcilvwl_655908owshcilvwl_655910owshcilvwl_655912owshcilvwl_655914owshcilvwl_655916*
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_6559052$
"owshcilvwl/StatefulPartitionedCallÆ
"oaettnoaty/StatefulPartitionedCallStatefulPartitionedCall+owshcilvwl/StatefulPartitionedCall:output:0oaettnoaty_655930oaettnoaty_655932*
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
F__inference_oaettnoaty_layer_call_and_return_conditional_losses_6559292$
"oaettnoaty/StatefulPartitionedCall
IdentityIdentity+oaettnoaty/StatefulPartitionedCall:output:0#^bpstkcuudk/StatefulPartitionedCall#^oaettnoaty/StatefulPartitionedCall#^osutmzfngz/StatefulPartitionedCall#^owshcilvwl/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"bpstkcuudk/StatefulPartitionedCall"bpstkcuudk/StatefulPartitionedCall2H
"oaettnoaty/StatefulPartitionedCall"oaettnoaty/StatefulPartitionedCall2H
"osutmzfngz/StatefulPartitionedCall"osutmzfngz/StatefulPartitionedCall2H
"owshcilvwl/StatefulPartitionedCall"owshcilvwl/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


å
while_cond_658584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_658584___redundant_placeholder04
0while_while_cond_658584___redundant_placeholder14
0while_while_cond_658584___redundant_placeholder24
0while_while_cond_658584___redundant_placeholder34
0while_while_cond_658584___redundant_placeholder44
0while_while_cond_658584___redundant_placeholder54
0while_while_cond_658584___redundant_placeholder6
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
+__inference_sequential_layer_call_fn_656577

jfowsgvbzw
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
jfowsgvbzwunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
F__inference_sequential_layer_call_and_return_conditional_losses_6565052
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
jfowsgvbzw
ßY

while_body_659125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_flzkvrshbq_matmul_readvariableop_resource_0:	 F
3while_flzkvrshbq_matmul_1_readvariableop_resource_0:	 A
2while_flzkvrshbq_biasadd_readvariableop_resource_0:	8
*while_flzkvrshbq_readvariableop_resource_0: :
,while_flzkvrshbq_readvariableop_1_resource_0: :
,while_flzkvrshbq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_flzkvrshbq_matmul_readvariableop_resource:	 D
1while_flzkvrshbq_matmul_1_readvariableop_resource:	 ?
0while_flzkvrshbq_biasadd_readvariableop_resource:	6
(while_flzkvrshbq_readvariableop_resource: 8
*while_flzkvrshbq_readvariableop_1_resource: 8
*while_flzkvrshbq_readvariableop_2_resource: ¢'while/flzkvrshbq/BiasAdd/ReadVariableOp¢&while/flzkvrshbq/MatMul/ReadVariableOp¢(while/flzkvrshbq/MatMul_1/ReadVariableOp¢while/flzkvrshbq/ReadVariableOp¢!while/flzkvrshbq/ReadVariableOp_1¢!while/flzkvrshbq/ReadVariableOp_2Ã
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
&while/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp1while_flzkvrshbq_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/flzkvrshbq/MatMul/ReadVariableOpÑ
while/flzkvrshbq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMulÉ
(while/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp3while_flzkvrshbq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/flzkvrshbq/MatMul_1/ReadVariableOpº
while/flzkvrshbq/MatMul_1MatMulwhile_placeholder_20while/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMul_1°
while/flzkvrshbq/addAddV2!while/flzkvrshbq/MatMul:product:0#while/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/addÂ
'while/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp2while_flzkvrshbq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/flzkvrshbq/BiasAdd/ReadVariableOp½
while/flzkvrshbq/BiasAddBiasAddwhile/flzkvrshbq/add:z:0/while/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/BiasAdd
 while/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/flzkvrshbq/split/split_dim
while/flzkvrshbq/splitSplit)while/flzkvrshbq/split/split_dim:output:0!while/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/flzkvrshbq/split©
while/flzkvrshbq/ReadVariableOpReadVariableOp*while_flzkvrshbq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/flzkvrshbq/ReadVariableOp£
while/flzkvrshbq/mulMul'while/flzkvrshbq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul¦
while/flzkvrshbq/add_1AddV2while/flzkvrshbq/split:output:0while/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_1
while/flzkvrshbq/SigmoidSigmoidwhile/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid¯
!while/flzkvrshbq/ReadVariableOp_1ReadVariableOp,while_flzkvrshbq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_1©
while/flzkvrshbq/mul_1Mul)while/flzkvrshbq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_1¨
while/flzkvrshbq/add_2AddV2while/flzkvrshbq/split:output:1while/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_2
while/flzkvrshbq/Sigmoid_1Sigmoidwhile/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_1
while/flzkvrshbq/mul_2Mulwhile/flzkvrshbq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_2
while/flzkvrshbq/TanhTanhwhile/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh¢
while/flzkvrshbq/mul_3Mulwhile/flzkvrshbq/Sigmoid:y:0while/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_3£
while/flzkvrshbq/add_3AddV2while/flzkvrshbq/mul_2:z:0while/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_3¯
!while/flzkvrshbq/ReadVariableOp_2ReadVariableOp,while_flzkvrshbq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_2°
while/flzkvrshbq/mul_4Mul)while/flzkvrshbq/ReadVariableOp_2:value:0while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_4¨
while/flzkvrshbq/add_4AddV2while/flzkvrshbq/split:output:3while/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_4
while/flzkvrshbq/Sigmoid_2Sigmoidwhile/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_2
while/flzkvrshbq/Tanh_1Tanhwhile/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh_1¦
while/flzkvrshbq/mul_5Mulwhile/flzkvrshbq/Sigmoid_2:y:0while/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/flzkvrshbq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/flzkvrshbq/mul_5:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/flzkvrshbq/add_3:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_flzkvrshbq_biasadd_readvariableop_resource2while_flzkvrshbq_biasadd_readvariableop_resource_0"h
1while_flzkvrshbq_matmul_1_readvariableop_resource3while_flzkvrshbq_matmul_1_readvariableop_resource_0"d
/while_flzkvrshbq_matmul_readvariableop_resource1while_flzkvrshbq_matmul_readvariableop_resource_0"Z
*while_flzkvrshbq_readvariableop_1_resource,while_flzkvrshbq_readvariableop_1_resource_0"Z
*while_flzkvrshbq_readvariableop_2_resource,while_flzkvrshbq_readvariableop_2_resource_0"V
(while_flzkvrshbq_readvariableop_resource*while_flzkvrshbq_readvariableop_resource_0")
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
'while/flzkvrshbq/BiasAdd/ReadVariableOp'while/flzkvrshbq/BiasAdd/ReadVariableOp2P
&while/flzkvrshbq/MatMul/ReadVariableOp&while/flzkvrshbq/MatMul/ReadVariableOp2T
(while/flzkvrshbq/MatMul_1/ReadVariableOp(while/flzkvrshbq/MatMul_1/ReadVariableOp2B
while/flzkvrshbq/ReadVariableOpwhile/flzkvrshbq/ReadVariableOp2F
!while/flzkvrshbq/ReadVariableOp_1!while/flzkvrshbq/ReadVariableOp_12F
!while/flzkvrshbq/ReadVariableOp_2!while/flzkvrshbq/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_654039

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
é

+__inference_owshcilvwl_layer_call_fn_658455
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_6548972
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
+__inference_osutmzfngz_layer_call_fn_657701

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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_6557122
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
ßY

while_body_658585
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_flzkvrshbq_matmul_readvariableop_resource_0:	 F
3while_flzkvrshbq_matmul_1_readvariableop_resource_0:	 A
2while_flzkvrshbq_biasadd_readvariableop_resource_0:	8
*while_flzkvrshbq_readvariableop_resource_0: :
,while_flzkvrshbq_readvariableop_1_resource_0: :
,while_flzkvrshbq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_flzkvrshbq_matmul_readvariableop_resource:	 D
1while_flzkvrshbq_matmul_1_readvariableop_resource:	 ?
0while_flzkvrshbq_biasadd_readvariableop_resource:	6
(while_flzkvrshbq_readvariableop_resource: 8
*while_flzkvrshbq_readvariableop_1_resource: 8
*while_flzkvrshbq_readvariableop_2_resource: ¢'while/flzkvrshbq/BiasAdd/ReadVariableOp¢&while/flzkvrshbq/MatMul/ReadVariableOp¢(while/flzkvrshbq/MatMul_1/ReadVariableOp¢while/flzkvrshbq/ReadVariableOp¢!while/flzkvrshbq/ReadVariableOp_1¢!while/flzkvrshbq/ReadVariableOp_2Ã
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
&while/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp1while_flzkvrshbq_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/flzkvrshbq/MatMul/ReadVariableOpÑ
while/flzkvrshbq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMulÉ
(while/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp3while_flzkvrshbq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/flzkvrshbq/MatMul_1/ReadVariableOpº
while/flzkvrshbq/MatMul_1MatMulwhile_placeholder_20while/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMul_1°
while/flzkvrshbq/addAddV2!while/flzkvrshbq/MatMul:product:0#while/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/addÂ
'while/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp2while_flzkvrshbq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/flzkvrshbq/BiasAdd/ReadVariableOp½
while/flzkvrshbq/BiasAddBiasAddwhile/flzkvrshbq/add:z:0/while/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/BiasAdd
 while/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/flzkvrshbq/split/split_dim
while/flzkvrshbq/splitSplit)while/flzkvrshbq/split/split_dim:output:0!while/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/flzkvrshbq/split©
while/flzkvrshbq/ReadVariableOpReadVariableOp*while_flzkvrshbq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/flzkvrshbq/ReadVariableOp£
while/flzkvrshbq/mulMul'while/flzkvrshbq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul¦
while/flzkvrshbq/add_1AddV2while/flzkvrshbq/split:output:0while/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_1
while/flzkvrshbq/SigmoidSigmoidwhile/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid¯
!while/flzkvrshbq/ReadVariableOp_1ReadVariableOp,while_flzkvrshbq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_1©
while/flzkvrshbq/mul_1Mul)while/flzkvrshbq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_1¨
while/flzkvrshbq/add_2AddV2while/flzkvrshbq/split:output:1while/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_2
while/flzkvrshbq/Sigmoid_1Sigmoidwhile/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_1
while/flzkvrshbq/mul_2Mulwhile/flzkvrshbq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_2
while/flzkvrshbq/TanhTanhwhile/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh¢
while/flzkvrshbq/mul_3Mulwhile/flzkvrshbq/Sigmoid:y:0while/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_3£
while/flzkvrshbq/add_3AddV2while/flzkvrshbq/mul_2:z:0while/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_3¯
!while/flzkvrshbq/ReadVariableOp_2ReadVariableOp,while_flzkvrshbq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_2°
while/flzkvrshbq/mul_4Mul)while/flzkvrshbq/ReadVariableOp_2:value:0while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_4¨
while/flzkvrshbq/add_4AddV2while/flzkvrshbq/split:output:3while/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_4
while/flzkvrshbq/Sigmoid_2Sigmoidwhile/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_2
while/flzkvrshbq/Tanh_1Tanhwhile/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh_1¦
while/flzkvrshbq/mul_5Mulwhile/flzkvrshbq/Sigmoid_2:y:0while/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/flzkvrshbq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/flzkvrshbq/mul_5:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/flzkvrshbq/add_3:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_flzkvrshbq_biasadd_readvariableop_resource2while_flzkvrshbq_biasadd_readvariableop_resource_0"h
1while_flzkvrshbq_matmul_1_readvariableop_resource3while_flzkvrshbq_matmul_1_readvariableop_resource_0"d
/while_flzkvrshbq_matmul_readvariableop_resource1while_flzkvrshbq_matmul_readvariableop_resource_0"Z
*while_flzkvrshbq_readvariableop_1_resource,while_flzkvrshbq_readvariableop_1_resource_0"Z
*while_flzkvrshbq_readvariableop_2_resource,while_flzkvrshbq_readvariableop_2_resource_0"V
(while_flzkvrshbq_readvariableop_resource*while_flzkvrshbq_readvariableop_resource_0")
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
'while/flzkvrshbq/BiasAdd/ReadVariableOp'while/flzkvrshbq/BiasAdd/ReadVariableOp2P
&while/flzkvrshbq/MatMul/ReadVariableOp&while/flzkvrshbq/MatMul/ReadVariableOp2T
(while/flzkvrshbq/MatMul_1/ReadVariableOp(while/flzkvrshbq/MatMul_1/ReadVariableOp2B
while/flzkvrshbq/ReadVariableOpwhile/flzkvrshbq/ReadVariableOp2F
!while/flzkvrshbq/ReadVariableOp_1!while/flzkvrshbq/ReadVariableOp_12F
!while/flzkvrshbq/ReadVariableOp_2!while/flzkvrshbq/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
Û
G
+__inference_xlcvyoxoxq_layer_call_fn_657637

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
F__inference_xlcvyoxoxq_layer_call_and_return_conditional_losses_6555312
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
°'
²
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_654226

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
£h

F__inference_owshcilvwl_layer_call_and_return_conditional_losses_656180

inputs<
)flzkvrshbq_matmul_readvariableop_resource:	 >
+flzkvrshbq_matmul_1_readvariableop_resource:	 9
*flzkvrshbq_biasadd_readvariableop_resource:	0
"flzkvrshbq_readvariableop_resource: 2
$flzkvrshbq_readvariableop_1_resource: 2
$flzkvrshbq_readvariableop_2_resource: 
identity¢!flzkvrshbq/BiasAdd/ReadVariableOp¢ flzkvrshbq/MatMul/ReadVariableOp¢"flzkvrshbq/MatMul_1/ReadVariableOp¢flzkvrshbq/ReadVariableOp¢flzkvrshbq/ReadVariableOp_1¢flzkvrshbq/ReadVariableOp_2¢whileD
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
 flzkvrshbq/MatMul/ReadVariableOpReadVariableOp)flzkvrshbq_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 flzkvrshbq/MatMul/ReadVariableOp§
flzkvrshbq/MatMulMatMulstrided_slice_2:output:0(flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMulµ
"flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp+flzkvrshbq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"flzkvrshbq/MatMul_1/ReadVariableOp£
flzkvrshbq/MatMul_1MatMulzeros:output:0*flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMul_1
flzkvrshbq/addAddV2flzkvrshbq/MatMul:product:0flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/add®
!flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp*flzkvrshbq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!flzkvrshbq/BiasAdd/ReadVariableOp¥
flzkvrshbq/BiasAddBiasAddflzkvrshbq/add:z:0)flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/BiasAddz
flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
flzkvrshbq/split/split_dimë
flzkvrshbq/splitSplit#flzkvrshbq/split/split_dim:output:0flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
flzkvrshbq/split
flzkvrshbq/ReadVariableOpReadVariableOp"flzkvrshbq_readvariableop_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp
flzkvrshbq/mulMul!flzkvrshbq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul
flzkvrshbq/add_1AddV2flzkvrshbq/split:output:0flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_1{
flzkvrshbq/SigmoidSigmoidflzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid
flzkvrshbq/ReadVariableOp_1ReadVariableOp$flzkvrshbq_readvariableop_1_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_1
flzkvrshbq/mul_1Mul#flzkvrshbq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_1
flzkvrshbq/add_2AddV2flzkvrshbq/split:output:1flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_2
flzkvrshbq/Sigmoid_1Sigmoidflzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_1
flzkvrshbq/mul_2Mulflzkvrshbq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_2w
flzkvrshbq/TanhTanhflzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh
flzkvrshbq/mul_3Mulflzkvrshbq/Sigmoid:y:0flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_3
flzkvrshbq/add_3AddV2flzkvrshbq/mul_2:z:0flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_3
flzkvrshbq/ReadVariableOp_2ReadVariableOp$flzkvrshbq_readvariableop_2_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_2
flzkvrshbq/mul_4Mul#flzkvrshbq/ReadVariableOp_2:value:0flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_4
flzkvrshbq/add_4AddV2flzkvrshbq/split:output:3flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_4
flzkvrshbq/Sigmoid_2Sigmoidflzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_2v
flzkvrshbq/Tanh_1Tanhflzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh_1
flzkvrshbq/mul_5Mulflzkvrshbq/Sigmoid_2:y:0flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)flzkvrshbq_matmul_readvariableop_resource+flzkvrshbq_matmul_1_readvariableop_resource*flzkvrshbq_biasadd_readvariableop_resource"flzkvrshbq_readvariableop_resource$flzkvrshbq_readvariableop_1_resource$flzkvrshbq_readvariableop_2_resource*
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
while_body_656079*
condR
while_cond_656078*Q
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
IdentityIdentitystrided_slice_3:output:0"^flzkvrshbq/BiasAdd/ReadVariableOp!^flzkvrshbq/MatMul/ReadVariableOp#^flzkvrshbq/MatMul_1/ReadVariableOp^flzkvrshbq/ReadVariableOp^flzkvrshbq/ReadVariableOp_1^flzkvrshbq/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!flzkvrshbq/BiasAdd/ReadVariableOp!flzkvrshbq/BiasAdd/ReadVariableOp2D
 flzkvrshbq/MatMul/ReadVariableOp flzkvrshbq/MatMul/ReadVariableOp2H
"flzkvrshbq/MatMul_1/ReadVariableOp"flzkvrshbq/MatMul_1/ReadVariableOp26
flzkvrshbq/ReadVariableOpflzkvrshbq/ReadVariableOp2:
flzkvrshbq/ReadVariableOp_1flzkvrshbq/ReadVariableOp_12:
flzkvrshbq/ReadVariableOp_2flzkvrshbq/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¸'
´
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_659513

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
F
ã
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_654139

inputs$
dsycfvoega_654040:	$
dsycfvoega_654042:	  
dsycfvoega_654044:	
dsycfvoega_654046: 
dsycfvoega_654048: 
dsycfvoega_654050: 
identity¢"dsycfvoega/StatefulPartitionedCall¢whileD
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
"dsycfvoega/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0dsycfvoega_654040dsycfvoega_654042dsycfvoega_654044dsycfvoega_654046dsycfvoega_654048dsycfvoega_654050*
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
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_6540392$
"dsycfvoega/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0dsycfvoega_654040dsycfvoega_654042dsycfvoega_654044dsycfvoega_654046dsycfvoega_654048dsycfvoega_654050*
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
while_body_654059*
condR
while_cond_654058*Q
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
IdentityIdentitytranspose_1:y:0#^dsycfvoega/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"dsycfvoega/StatefulPartitionedCall"dsycfvoega/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ö
!__inference__wrapped_model_653952

jfowsgvbzwW
Asequential_bpstkcuudk_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_bpstkcuudk_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_osutmzfngz_dsycfvoega_matmul_readvariableop_resource:	T
Asequential_osutmzfngz_dsycfvoega_matmul_1_readvariableop_resource:	 O
@sequential_osutmzfngz_dsycfvoega_biasadd_readvariableop_resource:	F
8sequential_osutmzfngz_dsycfvoega_readvariableop_resource: H
:sequential_osutmzfngz_dsycfvoega_readvariableop_1_resource: H
:sequential_osutmzfngz_dsycfvoega_readvariableop_2_resource: R
?sequential_owshcilvwl_flzkvrshbq_matmul_readvariableop_resource:	 T
Asequential_owshcilvwl_flzkvrshbq_matmul_1_readvariableop_resource:	 O
@sequential_owshcilvwl_flzkvrshbq_biasadd_readvariableop_resource:	F
8sequential_owshcilvwl_flzkvrshbq_readvariableop_resource: H
:sequential_owshcilvwl_flzkvrshbq_readvariableop_1_resource: H
:sequential_owshcilvwl_flzkvrshbq_readvariableop_2_resource: F
4sequential_oaettnoaty_matmul_readvariableop_resource: C
5sequential_oaettnoaty_biasadd_readvariableop_resource:
identity¢8sequential/bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,sequential/oaettnoaty/BiasAdd/ReadVariableOp¢+sequential/oaettnoaty/MatMul/ReadVariableOp¢7sequential/osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp¢6sequential/osutmzfngz/dsycfvoega/MatMul/ReadVariableOp¢8sequential/osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp¢/sequential/osutmzfngz/dsycfvoega/ReadVariableOp¢1sequential/osutmzfngz/dsycfvoega/ReadVariableOp_1¢1sequential/osutmzfngz/dsycfvoega/ReadVariableOp_2¢sequential/osutmzfngz/while¢7sequential/owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp¢6sequential/owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp¢8sequential/owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp¢/sequential/owshcilvwl/flzkvrshbq/ReadVariableOp¢1sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_1¢1sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_2¢sequential/owshcilvwl/while¥
+sequential/bpstkcuudk/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/bpstkcuudk/conv1d/ExpandDims/dimà
'sequential/bpstkcuudk/conv1d/ExpandDims
ExpandDims
jfowsgvbzw4sequential/bpstkcuudk/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/bpstkcuudk/conv1d/ExpandDimsú
8sequential/bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_bpstkcuudk_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/bpstkcuudk/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/bpstkcuudk/conv1d/ExpandDims_1/dim
)sequential/bpstkcuudk/conv1d/ExpandDims_1
ExpandDims@sequential/bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/bpstkcuudk/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/bpstkcuudk/conv1d/ExpandDims_1¨
"sequential/bpstkcuudk/conv1d/ShapeShape0sequential/bpstkcuudk/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/bpstkcuudk/conv1d/Shape®
0sequential/bpstkcuudk/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/bpstkcuudk/conv1d/strided_slice/stack»
2sequential/bpstkcuudk/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/bpstkcuudk/conv1d/strided_slice/stack_1²
2sequential/bpstkcuudk/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/bpstkcuudk/conv1d/strided_slice/stack_2
*sequential/bpstkcuudk/conv1d/strided_sliceStridedSlice+sequential/bpstkcuudk/conv1d/Shape:output:09sequential/bpstkcuudk/conv1d/strided_slice/stack:output:0;sequential/bpstkcuudk/conv1d/strided_slice/stack_1:output:0;sequential/bpstkcuudk/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/bpstkcuudk/conv1d/strided_slice±
*sequential/bpstkcuudk/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/bpstkcuudk/conv1d/Reshape/shapeø
$sequential/bpstkcuudk/conv1d/ReshapeReshape0sequential/bpstkcuudk/conv1d/ExpandDims:output:03sequential/bpstkcuudk/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/bpstkcuudk/conv1d/Reshape
#sequential/bpstkcuudk/conv1d/Conv2DConv2D-sequential/bpstkcuudk/conv1d/Reshape:output:02sequential/bpstkcuudk/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/bpstkcuudk/conv1d/Conv2D±
,sequential/bpstkcuudk/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/bpstkcuudk/conv1d/concat/values_1
(sequential/bpstkcuudk/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/bpstkcuudk/conv1d/concat/axis£
#sequential/bpstkcuudk/conv1d/concatConcatV23sequential/bpstkcuudk/conv1d/strided_slice:output:05sequential/bpstkcuudk/conv1d/concat/values_1:output:01sequential/bpstkcuudk/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/bpstkcuudk/conv1d/concatõ
&sequential/bpstkcuudk/conv1d/Reshape_1Reshape,sequential/bpstkcuudk/conv1d/Conv2D:output:0,sequential/bpstkcuudk/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/bpstkcuudk/conv1d/Reshape_1â
$sequential/bpstkcuudk/conv1d/SqueezeSqueeze/sequential/bpstkcuudk/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/bpstkcuudk/conv1d/Squeeze½
.sequential/bpstkcuudk/squeeze_batch_dims/ShapeShape-sequential/bpstkcuudk/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/bpstkcuudk/squeeze_batch_dims/ShapeÆ
<sequential/bpstkcuudk/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/bpstkcuudk/squeeze_batch_dims/strided_slice/stackÓ
>sequential/bpstkcuudk/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/bpstkcuudk/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/bpstkcuudk/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/bpstkcuudk/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/bpstkcuudk/squeeze_batch_dims/strided_sliceStridedSlice7sequential/bpstkcuudk/squeeze_batch_dims/Shape:output:0Esequential/bpstkcuudk/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/bpstkcuudk/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/bpstkcuudk/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/bpstkcuudk/squeeze_batch_dims/strided_sliceÅ
6sequential/bpstkcuudk/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/bpstkcuudk/squeeze_batch_dims/Reshape/shape
0sequential/bpstkcuudk/squeeze_batch_dims/ReshapeReshape-sequential/bpstkcuudk/conv1d/Squeeze:output:0?sequential/bpstkcuudk/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/bpstkcuudk/squeeze_batch_dims/Reshape
?sequential/bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_bpstkcuudk_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/bpstkcuudk/squeeze_batch_dims/BiasAddBiasAdd9sequential/bpstkcuudk/squeeze_batch_dims/Reshape:output:0Gsequential/bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/bpstkcuudk/squeeze_batch_dims/BiasAddÅ
8sequential/bpstkcuudk/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/bpstkcuudk/squeeze_batch_dims/concat/values_1·
4sequential/bpstkcuudk/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/bpstkcuudk/squeeze_batch_dims/concat/axisß
/sequential/bpstkcuudk/squeeze_batch_dims/concatConcatV2?sequential/bpstkcuudk/squeeze_batch_dims/strided_slice:output:0Asequential/bpstkcuudk/squeeze_batch_dims/concat/values_1:output:0=sequential/bpstkcuudk/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/bpstkcuudk/squeeze_batch_dims/concat¢
2sequential/bpstkcuudk/squeeze_batch_dims/Reshape_1Reshape9sequential/bpstkcuudk/squeeze_batch_dims/BiasAdd:output:08sequential/bpstkcuudk/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/bpstkcuudk/squeeze_batch_dims/Reshape_1¥
sequential/xlcvyoxoxq/ShapeShape;sequential/bpstkcuudk/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/xlcvyoxoxq/Shape 
)sequential/xlcvyoxoxq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/xlcvyoxoxq/strided_slice/stack¤
+sequential/xlcvyoxoxq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/xlcvyoxoxq/strided_slice/stack_1¤
+sequential/xlcvyoxoxq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/xlcvyoxoxq/strided_slice/stack_2æ
#sequential/xlcvyoxoxq/strided_sliceStridedSlice$sequential/xlcvyoxoxq/Shape:output:02sequential/xlcvyoxoxq/strided_slice/stack:output:04sequential/xlcvyoxoxq/strided_slice/stack_1:output:04sequential/xlcvyoxoxq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/xlcvyoxoxq/strided_slice
%sequential/xlcvyoxoxq/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/xlcvyoxoxq/Reshape/shape/1
%sequential/xlcvyoxoxq/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/xlcvyoxoxq/Reshape/shape/2
#sequential/xlcvyoxoxq/Reshape/shapePack,sequential/xlcvyoxoxq/strided_slice:output:0.sequential/xlcvyoxoxq/Reshape/shape/1:output:0.sequential/xlcvyoxoxq/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/xlcvyoxoxq/Reshape/shapeê
sequential/xlcvyoxoxq/ReshapeReshape;sequential/bpstkcuudk/squeeze_batch_dims/Reshape_1:output:0,sequential/xlcvyoxoxq/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/xlcvyoxoxq/Reshape
sequential/osutmzfngz/ShapeShape&sequential/xlcvyoxoxq/Reshape:output:0*
T0*
_output_shapes
:2
sequential/osutmzfngz/Shape 
)sequential/osutmzfngz/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/osutmzfngz/strided_slice/stack¤
+sequential/osutmzfngz/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/osutmzfngz/strided_slice/stack_1¤
+sequential/osutmzfngz/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/osutmzfngz/strided_slice/stack_2æ
#sequential/osutmzfngz/strided_sliceStridedSlice$sequential/osutmzfngz/Shape:output:02sequential/osutmzfngz/strided_slice/stack:output:04sequential/osutmzfngz/strided_slice/stack_1:output:04sequential/osutmzfngz/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/osutmzfngz/strided_slice
!sequential/osutmzfngz/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/osutmzfngz/zeros/mul/yÄ
sequential/osutmzfngz/zeros/mulMul,sequential/osutmzfngz/strided_slice:output:0*sequential/osutmzfngz/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/osutmzfngz/zeros/mul
"sequential/osutmzfngz/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/osutmzfngz/zeros/Less/y¿
 sequential/osutmzfngz/zeros/LessLess#sequential/osutmzfngz/zeros/mul:z:0+sequential/osutmzfngz/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/osutmzfngz/zeros/Less
$sequential/osutmzfngz/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/osutmzfngz/zeros/packed/1Û
"sequential/osutmzfngz/zeros/packedPack,sequential/osutmzfngz/strided_slice:output:0-sequential/osutmzfngz/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/osutmzfngz/zeros/packed
!sequential/osutmzfngz/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/osutmzfngz/zeros/ConstÍ
sequential/osutmzfngz/zerosFill+sequential/osutmzfngz/zeros/packed:output:0*sequential/osutmzfngz/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/osutmzfngz/zeros
#sequential/osutmzfngz/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/osutmzfngz/zeros_1/mul/yÊ
!sequential/osutmzfngz/zeros_1/mulMul,sequential/osutmzfngz/strided_slice:output:0,sequential/osutmzfngz/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/osutmzfngz/zeros_1/mul
$sequential/osutmzfngz/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/osutmzfngz/zeros_1/Less/yÇ
"sequential/osutmzfngz/zeros_1/LessLess%sequential/osutmzfngz/zeros_1/mul:z:0-sequential/osutmzfngz/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/osutmzfngz/zeros_1/Less
&sequential/osutmzfngz/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/osutmzfngz/zeros_1/packed/1á
$sequential/osutmzfngz/zeros_1/packedPack,sequential/osutmzfngz/strided_slice:output:0/sequential/osutmzfngz/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/osutmzfngz/zeros_1/packed
#sequential/osutmzfngz/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/osutmzfngz/zeros_1/ConstÕ
sequential/osutmzfngz/zeros_1Fill-sequential/osutmzfngz/zeros_1/packed:output:0,sequential/osutmzfngz/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/osutmzfngz/zeros_1¡
$sequential/osutmzfngz/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/osutmzfngz/transpose/permÜ
sequential/osutmzfngz/transpose	Transpose&sequential/xlcvyoxoxq/Reshape:output:0-sequential/osutmzfngz/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/osutmzfngz/transpose
sequential/osutmzfngz/Shape_1Shape#sequential/osutmzfngz/transpose:y:0*
T0*
_output_shapes
:2
sequential/osutmzfngz/Shape_1¤
+sequential/osutmzfngz/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/osutmzfngz/strided_slice_1/stack¨
-sequential/osutmzfngz/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/osutmzfngz/strided_slice_1/stack_1¨
-sequential/osutmzfngz/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/osutmzfngz/strided_slice_1/stack_2ò
%sequential/osutmzfngz/strided_slice_1StridedSlice&sequential/osutmzfngz/Shape_1:output:04sequential/osutmzfngz/strided_slice_1/stack:output:06sequential/osutmzfngz/strided_slice_1/stack_1:output:06sequential/osutmzfngz/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/osutmzfngz/strided_slice_1±
1sequential/osutmzfngz/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/osutmzfngz/TensorArrayV2/element_shape
#sequential/osutmzfngz/TensorArrayV2TensorListReserve:sequential/osutmzfngz/TensorArrayV2/element_shape:output:0.sequential/osutmzfngz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/osutmzfngz/TensorArrayV2ë
Ksequential/osutmzfngz/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/osutmzfngz/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/osutmzfngz/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/osutmzfngz/transpose:y:0Tsequential/osutmzfngz/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/osutmzfngz/TensorArrayUnstack/TensorListFromTensor¤
+sequential/osutmzfngz/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/osutmzfngz/strided_slice_2/stack¨
-sequential/osutmzfngz/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/osutmzfngz/strided_slice_2/stack_1¨
-sequential/osutmzfngz/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/osutmzfngz/strided_slice_2/stack_2
%sequential/osutmzfngz/strided_slice_2StridedSlice#sequential/osutmzfngz/transpose:y:04sequential/osutmzfngz/strided_slice_2/stack:output:06sequential/osutmzfngz/strided_slice_2/stack_1:output:06sequential/osutmzfngz/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/osutmzfngz/strided_slice_2ñ
6sequential/osutmzfngz/dsycfvoega/MatMul/ReadVariableOpReadVariableOp?sequential_osutmzfngz_dsycfvoega_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/osutmzfngz/dsycfvoega/MatMul/ReadVariableOpÿ
'sequential/osutmzfngz/dsycfvoega/MatMulMatMul.sequential/osutmzfngz/strided_slice_2:output:0>sequential/osutmzfngz/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/osutmzfngz/dsycfvoega/MatMul÷
8sequential/osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOpAsequential_osutmzfngz_dsycfvoega_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOpû
)sequential/osutmzfngz/dsycfvoega/MatMul_1MatMul$sequential/osutmzfngz/zeros:output:0@sequential/osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/osutmzfngz/dsycfvoega/MatMul_1ð
$sequential/osutmzfngz/dsycfvoega/addAddV21sequential/osutmzfngz/dsycfvoega/MatMul:product:03sequential/osutmzfngz/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/osutmzfngz/dsycfvoega/addð
7sequential/osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp@sequential_osutmzfngz_dsycfvoega_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOpý
(sequential/osutmzfngz/dsycfvoega/BiasAddBiasAdd(sequential/osutmzfngz/dsycfvoega/add:z:0?sequential/osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/osutmzfngz/dsycfvoega/BiasAdd¦
0sequential/osutmzfngz/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/osutmzfngz/dsycfvoega/split/split_dimÃ
&sequential/osutmzfngz/dsycfvoega/splitSplit9sequential/osutmzfngz/dsycfvoega/split/split_dim:output:01sequential/osutmzfngz/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/osutmzfngz/dsycfvoega/split×
/sequential/osutmzfngz/dsycfvoega/ReadVariableOpReadVariableOp8sequential_osutmzfngz_dsycfvoega_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/osutmzfngz/dsycfvoega/ReadVariableOpæ
$sequential/osutmzfngz/dsycfvoega/mulMul7sequential/osutmzfngz/dsycfvoega/ReadVariableOp:value:0&sequential/osutmzfngz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/osutmzfngz/dsycfvoega/mulæ
&sequential/osutmzfngz/dsycfvoega/add_1AddV2/sequential/osutmzfngz/dsycfvoega/split:output:0(sequential/osutmzfngz/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/dsycfvoega/add_1½
(sequential/osutmzfngz/dsycfvoega/SigmoidSigmoid*sequential/osutmzfngz/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/osutmzfngz/dsycfvoega/SigmoidÝ
1sequential/osutmzfngz/dsycfvoega/ReadVariableOp_1ReadVariableOp:sequential_osutmzfngz_dsycfvoega_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/osutmzfngz/dsycfvoega/ReadVariableOp_1ì
&sequential/osutmzfngz/dsycfvoega/mul_1Mul9sequential/osutmzfngz/dsycfvoega/ReadVariableOp_1:value:0&sequential/osutmzfngz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/dsycfvoega/mul_1è
&sequential/osutmzfngz/dsycfvoega/add_2AddV2/sequential/osutmzfngz/dsycfvoega/split:output:1*sequential/osutmzfngz/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/dsycfvoega/add_2Á
*sequential/osutmzfngz/dsycfvoega/Sigmoid_1Sigmoid*sequential/osutmzfngz/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/osutmzfngz/dsycfvoega/Sigmoid_1á
&sequential/osutmzfngz/dsycfvoega/mul_2Mul.sequential/osutmzfngz/dsycfvoega/Sigmoid_1:y:0&sequential/osutmzfngz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/dsycfvoega/mul_2¹
%sequential/osutmzfngz/dsycfvoega/TanhTanh/sequential/osutmzfngz/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/osutmzfngz/dsycfvoega/Tanhâ
&sequential/osutmzfngz/dsycfvoega/mul_3Mul,sequential/osutmzfngz/dsycfvoega/Sigmoid:y:0)sequential/osutmzfngz/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/dsycfvoega/mul_3ã
&sequential/osutmzfngz/dsycfvoega/add_3AddV2*sequential/osutmzfngz/dsycfvoega/mul_2:z:0*sequential/osutmzfngz/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/dsycfvoega/add_3Ý
1sequential/osutmzfngz/dsycfvoega/ReadVariableOp_2ReadVariableOp:sequential_osutmzfngz_dsycfvoega_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/osutmzfngz/dsycfvoega/ReadVariableOp_2ð
&sequential/osutmzfngz/dsycfvoega/mul_4Mul9sequential/osutmzfngz/dsycfvoega/ReadVariableOp_2:value:0*sequential/osutmzfngz/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/dsycfvoega/mul_4è
&sequential/osutmzfngz/dsycfvoega/add_4AddV2/sequential/osutmzfngz/dsycfvoega/split:output:3*sequential/osutmzfngz/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/dsycfvoega/add_4Á
*sequential/osutmzfngz/dsycfvoega/Sigmoid_2Sigmoid*sequential/osutmzfngz/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/osutmzfngz/dsycfvoega/Sigmoid_2¸
'sequential/osutmzfngz/dsycfvoega/Tanh_1Tanh*sequential/osutmzfngz/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/osutmzfngz/dsycfvoega/Tanh_1æ
&sequential/osutmzfngz/dsycfvoega/mul_5Mul.sequential/osutmzfngz/dsycfvoega/Sigmoid_2:y:0+sequential/osutmzfngz/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/dsycfvoega/mul_5»
3sequential/osutmzfngz/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/osutmzfngz/TensorArrayV2_1/element_shape
%sequential/osutmzfngz/TensorArrayV2_1TensorListReserve<sequential/osutmzfngz/TensorArrayV2_1/element_shape:output:0.sequential/osutmzfngz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/osutmzfngz/TensorArrayV2_1z
sequential/osutmzfngz/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/osutmzfngz/time«
.sequential/osutmzfngz/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/osutmzfngz/while/maximum_iterations
(sequential/osutmzfngz/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/osutmzfngz/while/loop_counterö	
sequential/osutmzfngz/whileWhile1sequential/osutmzfngz/while/loop_counter:output:07sequential/osutmzfngz/while/maximum_iterations:output:0#sequential/osutmzfngz/time:output:0.sequential/osutmzfngz/TensorArrayV2_1:handle:0$sequential/osutmzfngz/zeros:output:0&sequential/osutmzfngz/zeros_1:output:0.sequential/osutmzfngz/strided_slice_1:output:0Msequential/osutmzfngz/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_osutmzfngz_dsycfvoega_matmul_readvariableop_resourceAsequential_osutmzfngz_dsycfvoega_matmul_1_readvariableop_resource@sequential_osutmzfngz_dsycfvoega_biasadd_readvariableop_resource8sequential_osutmzfngz_dsycfvoega_readvariableop_resource:sequential_osutmzfngz_dsycfvoega_readvariableop_1_resource:sequential_osutmzfngz_dsycfvoega_readvariableop_2_resource*
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
'sequential_osutmzfngz_while_body_653669*3
cond+R)
'sequential_osutmzfngz_while_cond_653668*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/osutmzfngz/whileá
Fsequential/osutmzfngz/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/osutmzfngz/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/osutmzfngz/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/osutmzfngz/while:output:3Osequential/osutmzfngz/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/osutmzfngz/TensorArrayV2Stack/TensorListStack­
+sequential/osutmzfngz/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/osutmzfngz/strided_slice_3/stack¨
-sequential/osutmzfngz/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/osutmzfngz/strided_slice_3/stack_1¨
-sequential/osutmzfngz/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/osutmzfngz/strided_slice_3/stack_2
%sequential/osutmzfngz/strided_slice_3StridedSliceAsequential/osutmzfngz/TensorArrayV2Stack/TensorListStack:tensor:04sequential/osutmzfngz/strided_slice_3/stack:output:06sequential/osutmzfngz/strided_slice_3/stack_1:output:06sequential/osutmzfngz/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/osutmzfngz/strided_slice_3¥
&sequential/osutmzfngz/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/osutmzfngz/transpose_1/permý
!sequential/osutmzfngz/transpose_1	TransposeAsequential/osutmzfngz/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/osutmzfngz/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/osutmzfngz/transpose_1
sequential/owshcilvwl/ShapeShape%sequential/osutmzfngz/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/owshcilvwl/Shape 
)sequential/owshcilvwl/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/owshcilvwl/strided_slice/stack¤
+sequential/owshcilvwl/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/owshcilvwl/strided_slice/stack_1¤
+sequential/owshcilvwl/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/owshcilvwl/strided_slice/stack_2æ
#sequential/owshcilvwl/strided_sliceStridedSlice$sequential/owshcilvwl/Shape:output:02sequential/owshcilvwl/strided_slice/stack:output:04sequential/owshcilvwl/strided_slice/stack_1:output:04sequential/owshcilvwl/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/owshcilvwl/strided_slice
!sequential/owshcilvwl/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/owshcilvwl/zeros/mul/yÄ
sequential/owshcilvwl/zeros/mulMul,sequential/owshcilvwl/strided_slice:output:0*sequential/owshcilvwl/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/owshcilvwl/zeros/mul
"sequential/owshcilvwl/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/owshcilvwl/zeros/Less/y¿
 sequential/owshcilvwl/zeros/LessLess#sequential/owshcilvwl/zeros/mul:z:0+sequential/owshcilvwl/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/owshcilvwl/zeros/Less
$sequential/owshcilvwl/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/owshcilvwl/zeros/packed/1Û
"sequential/owshcilvwl/zeros/packedPack,sequential/owshcilvwl/strided_slice:output:0-sequential/owshcilvwl/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/owshcilvwl/zeros/packed
!sequential/owshcilvwl/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/owshcilvwl/zeros/ConstÍ
sequential/owshcilvwl/zerosFill+sequential/owshcilvwl/zeros/packed:output:0*sequential/owshcilvwl/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/owshcilvwl/zeros
#sequential/owshcilvwl/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/owshcilvwl/zeros_1/mul/yÊ
!sequential/owshcilvwl/zeros_1/mulMul,sequential/owshcilvwl/strided_slice:output:0,sequential/owshcilvwl/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/owshcilvwl/zeros_1/mul
$sequential/owshcilvwl/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/owshcilvwl/zeros_1/Less/yÇ
"sequential/owshcilvwl/zeros_1/LessLess%sequential/owshcilvwl/zeros_1/mul:z:0-sequential/owshcilvwl/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/owshcilvwl/zeros_1/Less
&sequential/owshcilvwl/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/owshcilvwl/zeros_1/packed/1á
$sequential/owshcilvwl/zeros_1/packedPack,sequential/owshcilvwl/strided_slice:output:0/sequential/owshcilvwl/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/owshcilvwl/zeros_1/packed
#sequential/owshcilvwl/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/owshcilvwl/zeros_1/ConstÕ
sequential/owshcilvwl/zeros_1Fill-sequential/owshcilvwl/zeros_1/packed:output:0,sequential/owshcilvwl/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/owshcilvwl/zeros_1¡
$sequential/owshcilvwl/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/owshcilvwl/transpose/permÛ
sequential/owshcilvwl/transpose	Transpose%sequential/osutmzfngz/transpose_1:y:0-sequential/owshcilvwl/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/owshcilvwl/transpose
sequential/owshcilvwl/Shape_1Shape#sequential/owshcilvwl/transpose:y:0*
T0*
_output_shapes
:2
sequential/owshcilvwl/Shape_1¤
+sequential/owshcilvwl/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/owshcilvwl/strided_slice_1/stack¨
-sequential/owshcilvwl/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/owshcilvwl/strided_slice_1/stack_1¨
-sequential/owshcilvwl/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/owshcilvwl/strided_slice_1/stack_2ò
%sequential/owshcilvwl/strided_slice_1StridedSlice&sequential/owshcilvwl/Shape_1:output:04sequential/owshcilvwl/strided_slice_1/stack:output:06sequential/owshcilvwl/strided_slice_1/stack_1:output:06sequential/owshcilvwl/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/owshcilvwl/strided_slice_1±
1sequential/owshcilvwl/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/owshcilvwl/TensorArrayV2/element_shape
#sequential/owshcilvwl/TensorArrayV2TensorListReserve:sequential/owshcilvwl/TensorArrayV2/element_shape:output:0.sequential/owshcilvwl/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/owshcilvwl/TensorArrayV2ë
Ksequential/owshcilvwl/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/owshcilvwl/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/owshcilvwl/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/owshcilvwl/transpose:y:0Tsequential/owshcilvwl/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/owshcilvwl/TensorArrayUnstack/TensorListFromTensor¤
+sequential/owshcilvwl/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/owshcilvwl/strided_slice_2/stack¨
-sequential/owshcilvwl/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/owshcilvwl/strided_slice_2/stack_1¨
-sequential/owshcilvwl/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/owshcilvwl/strided_slice_2/stack_2
%sequential/owshcilvwl/strided_slice_2StridedSlice#sequential/owshcilvwl/transpose:y:04sequential/owshcilvwl/strided_slice_2/stack:output:06sequential/owshcilvwl/strided_slice_2/stack_1:output:06sequential/owshcilvwl/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/owshcilvwl/strided_slice_2ñ
6sequential/owshcilvwl/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp?sequential_owshcilvwl_flzkvrshbq_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/owshcilvwl/flzkvrshbq/MatMul/ReadVariableOpÿ
'sequential/owshcilvwl/flzkvrshbq/MatMulMatMul.sequential/owshcilvwl/strided_slice_2:output:0>sequential/owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/owshcilvwl/flzkvrshbq/MatMul÷
8sequential/owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOpAsequential_owshcilvwl_flzkvrshbq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOpû
)sequential/owshcilvwl/flzkvrshbq/MatMul_1MatMul$sequential/owshcilvwl/zeros:output:0@sequential/owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/owshcilvwl/flzkvrshbq/MatMul_1ð
$sequential/owshcilvwl/flzkvrshbq/addAddV21sequential/owshcilvwl/flzkvrshbq/MatMul:product:03sequential/owshcilvwl/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/owshcilvwl/flzkvrshbq/addð
7sequential/owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp@sequential_owshcilvwl_flzkvrshbq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOpý
(sequential/owshcilvwl/flzkvrshbq/BiasAddBiasAdd(sequential/owshcilvwl/flzkvrshbq/add:z:0?sequential/owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/owshcilvwl/flzkvrshbq/BiasAdd¦
0sequential/owshcilvwl/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/owshcilvwl/flzkvrshbq/split/split_dimÃ
&sequential/owshcilvwl/flzkvrshbq/splitSplit9sequential/owshcilvwl/flzkvrshbq/split/split_dim:output:01sequential/owshcilvwl/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/owshcilvwl/flzkvrshbq/split×
/sequential/owshcilvwl/flzkvrshbq/ReadVariableOpReadVariableOp8sequential_owshcilvwl_flzkvrshbq_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/owshcilvwl/flzkvrshbq/ReadVariableOpæ
$sequential/owshcilvwl/flzkvrshbq/mulMul7sequential/owshcilvwl/flzkvrshbq/ReadVariableOp:value:0&sequential/owshcilvwl/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/owshcilvwl/flzkvrshbq/mulæ
&sequential/owshcilvwl/flzkvrshbq/add_1AddV2/sequential/owshcilvwl/flzkvrshbq/split:output:0(sequential/owshcilvwl/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/flzkvrshbq/add_1½
(sequential/owshcilvwl/flzkvrshbq/SigmoidSigmoid*sequential/owshcilvwl/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/owshcilvwl/flzkvrshbq/SigmoidÝ
1sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_1ReadVariableOp:sequential_owshcilvwl_flzkvrshbq_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_1ì
&sequential/owshcilvwl/flzkvrshbq/mul_1Mul9sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_1:value:0&sequential/owshcilvwl/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/flzkvrshbq/mul_1è
&sequential/owshcilvwl/flzkvrshbq/add_2AddV2/sequential/owshcilvwl/flzkvrshbq/split:output:1*sequential/owshcilvwl/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/flzkvrshbq/add_2Á
*sequential/owshcilvwl/flzkvrshbq/Sigmoid_1Sigmoid*sequential/owshcilvwl/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/owshcilvwl/flzkvrshbq/Sigmoid_1á
&sequential/owshcilvwl/flzkvrshbq/mul_2Mul.sequential/owshcilvwl/flzkvrshbq/Sigmoid_1:y:0&sequential/owshcilvwl/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/flzkvrshbq/mul_2¹
%sequential/owshcilvwl/flzkvrshbq/TanhTanh/sequential/owshcilvwl/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/owshcilvwl/flzkvrshbq/Tanhâ
&sequential/owshcilvwl/flzkvrshbq/mul_3Mul,sequential/owshcilvwl/flzkvrshbq/Sigmoid:y:0)sequential/owshcilvwl/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/flzkvrshbq/mul_3ã
&sequential/owshcilvwl/flzkvrshbq/add_3AddV2*sequential/owshcilvwl/flzkvrshbq/mul_2:z:0*sequential/owshcilvwl/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/flzkvrshbq/add_3Ý
1sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_2ReadVariableOp:sequential_owshcilvwl_flzkvrshbq_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_2ð
&sequential/owshcilvwl/flzkvrshbq/mul_4Mul9sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_2:value:0*sequential/owshcilvwl/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/flzkvrshbq/mul_4è
&sequential/owshcilvwl/flzkvrshbq/add_4AddV2/sequential/owshcilvwl/flzkvrshbq/split:output:3*sequential/owshcilvwl/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/flzkvrshbq/add_4Á
*sequential/owshcilvwl/flzkvrshbq/Sigmoid_2Sigmoid*sequential/owshcilvwl/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/owshcilvwl/flzkvrshbq/Sigmoid_2¸
'sequential/owshcilvwl/flzkvrshbq/Tanh_1Tanh*sequential/owshcilvwl/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/owshcilvwl/flzkvrshbq/Tanh_1æ
&sequential/owshcilvwl/flzkvrshbq/mul_5Mul.sequential/owshcilvwl/flzkvrshbq/Sigmoid_2:y:0+sequential/owshcilvwl/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/owshcilvwl/flzkvrshbq/mul_5»
3sequential/owshcilvwl/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/owshcilvwl/TensorArrayV2_1/element_shape
%sequential/owshcilvwl/TensorArrayV2_1TensorListReserve<sequential/owshcilvwl/TensorArrayV2_1/element_shape:output:0.sequential/owshcilvwl/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/owshcilvwl/TensorArrayV2_1z
sequential/owshcilvwl/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/owshcilvwl/time«
.sequential/owshcilvwl/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/owshcilvwl/while/maximum_iterations
(sequential/owshcilvwl/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/owshcilvwl/while/loop_counterö	
sequential/owshcilvwl/whileWhile1sequential/owshcilvwl/while/loop_counter:output:07sequential/owshcilvwl/while/maximum_iterations:output:0#sequential/owshcilvwl/time:output:0.sequential/owshcilvwl/TensorArrayV2_1:handle:0$sequential/owshcilvwl/zeros:output:0&sequential/owshcilvwl/zeros_1:output:0.sequential/owshcilvwl/strided_slice_1:output:0Msequential/owshcilvwl/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_owshcilvwl_flzkvrshbq_matmul_readvariableop_resourceAsequential_owshcilvwl_flzkvrshbq_matmul_1_readvariableop_resource@sequential_owshcilvwl_flzkvrshbq_biasadd_readvariableop_resource8sequential_owshcilvwl_flzkvrshbq_readvariableop_resource:sequential_owshcilvwl_flzkvrshbq_readvariableop_1_resource:sequential_owshcilvwl_flzkvrshbq_readvariableop_2_resource*
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
'sequential_owshcilvwl_while_body_653845*3
cond+R)
'sequential_owshcilvwl_while_cond_653844*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/owshcilvwl/whileá
Fsequential/owshcilvwl/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/owshcilvwl/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/owshcilvwl/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/owshcilvwl/while:output:3Osequential/owshcilvwl/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/owshcilvwl/TensorArrayV2Stack/TensorListStack­
+sequential/owshcilvwl/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/owshcilvwl/strided_slice_3/stack¨
-sequential/owshcilvwl/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/owshcilvwl/strided_slice_3/stack_1¨
-sequential/owshcilvwl/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/owshcilvwl/strided_slice_3/stack_2
%sequential/owshcilvwl/strided_slice_3StridedSliceAsequential/owshcilvwl/TensorArrayV2Stack/TensorListStack:tensor:04sequential/owshcilvwl/strided_slice_3/stack:output:06sequential/owshcilvwl/strided_slice_3/stack_1:output:06sequential/owshcilvwl/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/owshcilvwl/strided_slice_3¥
&sequential/owshcilvwl/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/owshcilvwl/transpose_1/permý
!sequential/owshcilvwl/transpose_1	TransposeAsequential/owshcilvwl/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/owshcilvwl/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/owshcilvwl/transpose_1Ï
+sequential/oaettnoaty/MatMul/ReadVariableOpReadVariableOp4sequential_oaettnoaty_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/oaettnoaty/MatMul/ReadVariableOpÝ
sequential/oaettnoaty/MatMulMatMul.sequential/owshcilvwl/strided_slice_3:output:03sequential/oaettnoaty/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/oaettnoaty/MatMulÎ
,sequential/oaettnoaty/BiasAdd/ReadVariableOpReadVariableOp5sequential_oaettnoaty_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/oaettnoaty/BiasAdd/ReadVariableOpÙ
sequential/oaettnoaty/BiasAddBiasAdd&sequential/oaettnoaty/MatMul:product:04sequential/oaettnoaty/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/oaettnoaty/BiasAdd 
IdentityIdentity&sequential/oaettnoaty/BiasAdd:output:09^sequential/bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp@^sequential/bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp-^sequential/oaettnoaty/BiasAdd/ReadVariableOp,^sequential/oaettnoaty/MatMul/ReadVariableOp8^sequential/osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp7^sequential/osutmzfngz/dsycfvoega/MatMul/ReadVariableOp9^sequential/osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp0^sequential/osutmzfngz/dsycfvoega/ReadVariableOp2^sequential/osutmzfngz/dsycfvoega/ReadVariableOp_12^sequential/osutmzfngz/dsycfvoega/ReadVariableOp_2^sequential/osutmzfngz/while8^sequential/owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp7^sequential/owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp9^sequential/owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp0^sequential/owshcilvwl/flzkvrshbq/ReadVariableOp2^sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_12^sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_2^sequential/owshcilvwl/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2t
8sequential/bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp8sequential/bpstkcuudk/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/bpstkcuudk/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,sequential/oaettnoaty/BiasAdd/ReadVariableOp,sequential/oaettnoaty/BiasAdd/ReadVariableOp2Z
+sequential/oaettnoaty/MatMul/ReadVariableOp+sequential/oaettnoaty/MatMul/ReadVariableOp2r
7sequential/osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp7sequential/osutmzfngz/dsycfvoega/BiasAdd/ReadVariableOp2p
6sequential/osutmzfngz/dsycfvoega/MatMul/ReadVariableOp6sequential/osutmzfngz/dsycfvoega/MatMul/ReadVariableOp2t
8sequential/osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp8sequential/osutmzfngz/dsycfvoega/MatMul_1/ReadVariableOp2b
/sequential/osutmzfngz/dsycfvoega/ReadVariableOp/sequential/osutmzfngz/dsycfvoega/ReadVariableOp2f
1sequential/osutmzfngz/dsycfvoega/ReadVariableOp_11sequential/osutmzfngz/dsycfvoega/ReadVariableOp_12f
1sequential/osutmzfngz/dsycfvoega/ReadVariableOp_21sequential/osutmzfngz/dsycfvoega/ReadVariableOp_22:
sequential/osutmzfngz/whilesequential/osutmzfngz/while2r
7sequential/owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp7sequential/owshcilvwl/flzkvrshbq/BiasAdd/ReadVariableOp2p
6sequential/owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp6sequential/owshcilvwl/flzkvrshbq/MatMul/ReadVariableOp2t
8sequential/owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp8sequential/owshcilvwl/flzkvrshbq/MatMul_1/ReadVariableOp2b
/sequential/owshcilvwl/flzkvrshbq/ReadVariableOp/sequential/owshcilvwl/flzkvrshbq/ReadVariableOp2f
1sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_11sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_12f
1sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_21sequential/owshcilvwl/flzkvrshbq/ReadVariableOp_22:
sequential/owshcilvwl/whilesequential/owshcilvwl/while:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
jfowsgvbzw


å
while_cond_658336
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_658336___redundant_placeholder04
0while_while_cond_658336___redundant_placeholder14
0while_while_cond_658336___redundant_placeholder24
0while_while_cond_658336___redundant_placeholder34
0while_while_cond_658336___redundant_placeholder44
0while_while_cond_658336___redundant_placeholder54
0while_while_cond_658336___redundant_placeholder6
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
while_cond_655610
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_655610___redundant_placeholder04
0while_while_cond_655610___redundant_placeholder14
0while_while_cond_655610___redundant_placeholder24
0while_while_cond_655610___redundant_placeholder34
0while_while_cond_655610___redundant_placeholder44
0while_while_cond_655610___redundant_placeholder54
0while_while_cond_655610___redundant_placeholder6
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
while_body_658337
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dsycfvoega_matmul_readvariableop_resource_0:	F
3while_dsycfvoega_matmul_1_readvariableop_resource_0:	 A
2while_dsycfvoega_biasadd_readvariableop_resource_0:	8
*while_dsycfvoega_readvariableop_resource_0: :
,while_dsycfvoega_readvariableop_1_resource_0: :
,while_dsycfvoega_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dsycfvoega_matmul_readvariableop_resource:	D
1while_dsycfvoega_matmul_1_readvariableop_resource:	 ?
0while_dsycfvoega_biasadd_readvariableop_resource:	6
(while_dsycfvoega_readvariableop_resource: 8
*while_dsycfvoega_readvariableop_1_resource: 8
*while_dsycfvoega_readvariableop_2_resource: ¢'while/dsycfvoega/BiasAdd/ReadVariableOp¢&while/dsycfvoega/MatMul/ReadVariableOp¢(while/dsycfvoega/MatMul_1/ReadVariableOp¢while/dsycfvoega/ReadVariableOp¢!while/dsycfvoega/ReadVariableOp_1¢!while/dsycfvoega/ReadVariableOp_2Ã
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
&while/dsycfvoega/MatMul/ReadVariableOpReadVariableOp1while_dsycfvoega_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dsycfvoega/MatMul/ReadVariableOpÑ
while/dsycfvoega/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMulÉ
(while/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp3while_dsycfvoega_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dsycfvoega/MatMul_1/ReadVariableOpº
while/dsycfvoega/MatMul_1MatMulwhile_placeholder_20while/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMul_1°
while/dsycfvoega/addAddV2!while/dsycfvoega/MatMul:product:0#while/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/addÂ
'while/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp2while_dsycfvoega_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dsycfvoega/BiasAdd/ReadVariableOp½
while/dsycfvoega/BiasAddBiasAddwhile/dsycfvoega/add:z:0/while/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/BiasAdd
 while/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dsycfvoega/split/split_dim
while/dsycfvoega/splitSplit)while/dsycfvoega/split/split_dim:output:0!while/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dsycfvoega/split©
while/dsycfvoega/ReadVariableOpReadVariableOp*while_dsycfvoega_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dsycfvoega/ReadVariableOp£
while/dsycfvoega/mulMul'while/dsycfvoega/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul¦
while/dsycfvoega/add_1AddV2while/dsycfvoega/split:output:0while/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_1
while/dsycfvoega/SigmoidSigmoidwhile/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid¯
!while/dsycfvoega/ReadVariableOp_1ReadVariableOp,while_dsycfvoega_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_1©
while/dsycfvoega/mul_1Mul)while/dsycfvoega/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_1¨
while/dsycfvoega/add_2AddV2while/dsycfvoega/split:output:1while/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_2
while/dsycfvoega/Sigmoid_1Sigmoidwhile/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_1
while/dsycfvoega/mul_2Mulwhile/dsycfvoega/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_2
while/dsycfvoega/TanhTanhwhile/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh¢
while/dsycfvoega/mul_3Mulwhile/dsycfvoega/Sigmoid:y:0while/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_3£
while/dsycfvoega/add_3AddV2while/dsycfvoega/mul_2:z:0while/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_3¯
!while/dsycfvoega/ReadVariableOp_2ReadVariableOp,while_dsycfvoega_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_2°
while/dsycfvoega/mul_4Mul)while/dsycfvoega/ReadVariableOp_2:value:0while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_4¨
while/dsycfvoega/add_4AddV2while/dsycfvoega/split:output:3while/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_4
while/dsycfvoega/Sigmoid_2Sigmoidwhile/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_2
while/dsycfvoega/Tanh_1Tanhwhile/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh_1¦
while/dsycfvoega/mul_5Mulwhile/dsycfvoega/Sigmoid_2:y:0while/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dsycfvoega/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dsycfvoega/mul_5:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dsycfvoega/add_3:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dsycfvoega_biasadd_readvariableop_resource2while_dsycfvoega_biasadd_readvariableop_resource_0"h
1while_dsycfvoega_matmul_1_readvariableop_resource3while_dsycfvoega_matmul_1_readvariableop_resource_0"d
/while_dsycfvoega_matmul_readvariableop_resource1while_dsycfvoega_matmul_readvariableop_resource_0"Z
*while_dsycfvoega_readvariableop_1_resource,while_dsycfvoega_readvariableop_1_resource_0"Z
*while_dsycfvoega_readvariableop_2_resource,while_dsycfvoega_readvariableop_2_resource_0"V
(while_dsycfvoega_readvariableop_resource*while_dsycfvoega_readvariableop_resource_0")
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
'while/dsycfvoega/BiasAdd/ReadVariableOp'while/dsycfvoega/BiasAdd/ReadVariableOp2P
&while/dsycfvoega/MatMul/ReadVariableOp&while/dsycfvoega/MatMul/ReadVariableOp2T
(while/dsycfvoega/MatMul_1/ReadVariableOp(while/dsycfvoega/MatMul_1/ReadVariableOp2B
while/dsycfvoega/ReadVariableOpwhile/dsycfvoega/ReadVariableOp2F
!while/dsycfvoega/ReadVariableOp_1!while/dsycfvoega/ReadVariableOp_12F
!while/dsycfvoega/ReadVariableOp_2!while/dsycfvoega/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_655611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dsycfvoega_matmul_readvariableop_resource_0:	F
3while_dsycfvoega_matmul_1_readvariableop_resource_0:	 A
2while_dsycfvoega_biasadd_readvariableop_resource_0:	8
*while_dsycfvoega_readvariableop_resource_0: :
,while_dsycfvoega_readvariableop_1_resource_0: :
,while_dsycfvoega_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dsycfvoega_matmul_readvariableop_resource:	D
1while_dsycfvoega_matmul_1_readvariableop_resource:	 ?
0while_dsycfvoega_biasadd_readvariableop_resource:	6
(while_dsycfvoega_readvariableop_resource: 8
*while_dsycfvoega_readvariableop_1_resource: 8
*while_dsycfvoega_readvariableop_2_resource: ¢'while/dsycfvoega/BiasAdd/ReadVariableOp¢&while/dsycfvoega/MatMul/ReadVariableOp¢(while/dsycfvoega/MatMul_1/ReadVariableOp¢while/dsycfvoega/ReadVariableOp¢!while/dsycfvoega/ReadVariableOp_1¢!while/dsycfvoega/ReadVariableOp_2Ã
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
&while/dsycfvoega/MatMul/ReadVariableOpReadVariableOp1while_dsycfvoega_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dsycfvoega/MatMul/ReadVariableOpÑ
while/dsycfvoega/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMulÉ
(while/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp3while_dsycfvoega_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dsycfvoega/MatMul_1/ReadVariableOpº
while/dsycfvoega/MatMul_1MatMulwhile_placeholder_20while/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMul_1°
while/dsycfvoega/addAddV2!while/dsycfvoega/MatMul:product:0#while/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/addÂ
'while/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp2while_dsycfvoega_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dsycfvoega/BiasAdd/ReadVariableOp½
while/dsycfvoega/BiasAddBiasAddwhile/dsycfvoega/add:z:0/while/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/BiasAdd
 while/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dsycfvoega/split/split_dim
while/dsycfvoega/splitSplit)while/dsycfvoega/split/split_dim:output:0!while/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dsycfvoega/split©
while/dsycfvoega/ReadVariableOpReadVariableOp*while_dsycfvoega_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dsycfvoega/ReadVariableOp£
while/dsycfvoega/mulMul'while/dsycfvoega/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul¦
while/dsycfvoega/add_1AddV2while/dsycfvoega/split:output:0while/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_1
while/dsycfvoega/SigmoidSigmoidwhile/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid¯
!while/dsycfvoega/ReadVariableOp_1ReadVariableOp,while_dsycfvoega_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_1©
while/dsycfvoega/mul_1Mul)while/dsycfvoega/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_1¨
while/dsycfvoega/add_2AddV2while/dsycfvoega/split:output:1while/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_2
while/dsycfvoega/Sigmoid_1Sigmoidwhile/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_1
while/dsycfvoega/mul_2Mulwhile/dsycfvoega/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_2
while/dsycfvoega/TanhTanhwhile/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh¢
while/dsycfvoega/mul_3Mulwhile/dsycfvoega/Sigmoid:y:0while/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_3£
while/dsycfvoega/add_3AddV2while/dsycfvoega/mul_2:z:0while/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_3¯
!while/dsycfvoega/ReadVariableOp_2ReadVariableOp,while_dsycfvoega_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_2°
while/dsycfvoega/mul_4Mul)while/dsycfvoega/ReadVariableOp_2:value:0while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_4¨
while/dsycfvoega/add_4AddV2while/dsycfvoega/split:output:3while/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_4
while/dsycfvoega/Sigmoid_2Sigmoidwhile/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_2
while/dsycfvoega/Tanh_1Tanhwhile/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh_1¦
while/dsycfvoega/mul_5Mulwhile/dsycfvoega/Sigmoid_2:y:0while/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dsycfvoega/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dsycfvoega/mul_5:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dsycfvoega/add_3:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dsycfvoega_biasadd_readvariableop_resource2while_dsycfvoega_biasadd_readvariableop_resource_0"h
1while_dsycfvoega_matmul_1_readvariableop_resource3while_dsycfvoega_matmul_1_readvariableop_resource_0"d
/while_dsycfvoega_matmul_readvariableop_resource1while_dsycfvoega_matmul_readvariableop_resource_0"Z
*while_dsycfvoega_readvariableop_1_resource,while_dsycfvoega_readvariableop_1_resource_0"Z
*while_dsycfvoega_readvariableop_2_resource,while_dsycfvoega_readvariableop_2_resource_0"V
(while_dsycfvoega_readvariableop_resource*while_dsycfvoega_readvariableop_resource_0")
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
'while/dsycfvoega/BiasAdd/ReadVariableOp'while/dsycfvoega/BiasAdd/ReadVariableOp2P
&while/dsycfvoega/MatMul/ReadVariableOp&while/dsycfvoega/MatMul/ReadVariableOp2T
(while/dsycfvoega/MatMul_1/ReadVariableOp(while/dsycfvoega/MatMul_1/ReadVariableOp2B
while/dsycfvoega/ReadVariableOpwhile/dsycfvoega/ReadVariableOp2F
!while/dsycfvoega/ReadVariableOp_1!while/dsycfvoega/ReadVariableOp_12F
!while/dsycfvoega/ReadVariableOp_2!while/dsycfvoega/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
+__inference_owshcilvwl_layer_call_fn_658472
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_6551602
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
F
ã
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_655160

inputs$
flzkvrshbq_655061:	 $
flzkvrshbq_655063:	  
flzkvrshbq_655065:	
flzkvrshbq_655067: 
flzkvrshbq_655069: 
flzkvrshbq_655071: 
identity¢"flzkvrshbq/StatefulPartitionedCall¢whileD
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
"flzkvrshbq/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0flzkvrshbq_655061flzkvrshbq_655063flzkvrshbq_655065flzkvrshbq_655067flzkvrshbq_655069flzkvrshbq_655071*
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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_6549842$
"flzkvrshbq/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0flzkvrshbq_655061flzkvrshbq_655063flzkvrshbq_655065flzkvrshbq_655067flzkvrshbq_655069flzkvrshbq_655071*
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
while_body_655080*
condR
while_cond_655079*Q
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
IdentityIdentitystrided_slice_3:output:0#^flzkvrshbq/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"flzkvrshbq/StatefulPartitionedCall"flzkvrshbq/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


å
while_cond_654321
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_654321___redundant_placeholder04
0while_while_cond_654321___redundant_placeholder14
0while_while_cond_654321___redundant_placeholder24
0while_while_cond_654321___redundant_placeholder34
0while_while_cond_654321___redundant_placeholder44
0while_while_cond_654321___redundant_placeholder54
0while_while_cond_654321___redundant_placeholder6
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
while_cond_656292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_656292___redundant_placeholder04
0while_while_cond_656292___redundant_placeholder14
0while_while_cond_656292___redundant_placeholder24
0while_while_cond_656292___redundant_placeholder34
0while_while_cond_656292___redundant_placeholder44
0while_while_cond_656292___redundant_placeholder54
0while_while_cond_656292___redundant_placeholder6
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
Û

'sequential_osutmzfngz_while_body_653669H
Dsequential_osutmzfngz_while_sequential_osutmzfngz_while_loop_counterN
Jsequential_osutmzfngz_while_sequential_osutmzfngz_while_maximum_iterations+
'sequential_osutmzfngz_while_placeholder-
)sequential_osutmzfngz_while_placeholder_1-
)sequential_osutmzfngz_while_placeholder_2-
)sequential_osutmzfngz_while_placeholder_3G
Csequential_osutmzfngz_while_sequential_osutmzfngz_strided_slice_1_0
sequential_osutmzfngz_while_tensorarrayv2read_tensorlistgetitem_sequential_osutmzfngz_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource_0:	\
Isequential_osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource_0:	 W
Hsequential_osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource_0:	N
@sequential_osutmzfngz_while_dsycfvoega_readvariableop_resource_0: P
Bsequential_osutmzfngz_while_dsycfvoega_readvariableop_1_resource_0: P
Bsequential_osutmzfngz_while_dsycfvoega_readvariableop_2_resource_0: (
$sequential_osutmzfngz_while_identity*
&sequential_osutmzfngz_while_identity_1*
&sequential_osutmzfngz_while_identity_2*
&sequential_osutmzfngz_while_identity_3*
&sequential_osutmzfngz_while_identity_4*
&sequential_osutmzfngz_while_identity_5E
Asequential_osutmzfngz_while_sequential_osutmzfngz_strided_slice_1
}sequential_osutmzfngz_while_tensorarrayv2read_tensorlistgetitem_sequential_osutmzfngz_tensorarrayunstack_tensorlistfromtensorX
Esequential_osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource:	Z
Gsequential_osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource:	 U
Fsequential_osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource:	L
>sequential_osutmzfngz_while_dsycfvoega_readvariableop_resource: N
@sequential_osutmzfngz_while_dsycfvoega_readvariableop_1_resource: N
@sequential_osutmzfngz_while_dsycfvoega_readvariableop_2_resource: ¢=sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp¢<sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp¢>sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp¢5sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp¢7sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_1¢7sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2ï
Msequential/osutmzfngz/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/osutmzfngz/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/osutmzfngz/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_osutmzfngz_while_tensorarrayv2read_tensorlistgetitem_sequential_osutmzfngz_tensorarrayunstack_tensorlistfromtensor_0'sequential_osutmzfngz_while_placeholderVsequential/osutmzfngz/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/osutmzfngz/while/TensorArrayV2Read/TensorListGetItem
<sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOpReadVariableOpGsequential_osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp©
-sequential/osutmzfngz/while/dsycfvoega/MatMulMatMulFsequential/osutmzfngz/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/osutmzfngz/while/dsycfvoega/MatMul
>sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOpIsequential_osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp
/sequential/osutmzfngz/while/dsycfvoega/MatMul_1MatMul)sequential_osutmzfngz_while_placeholder_2Fsequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/osutmzfngz/while/dsycfvoega/MatMul_1
*sequential/osutmzfngz/while/dsycfvoega/addAddV27sequential/osutmzfngz/while/dsycfvoega/MatMul:product:09sequential/osutmzfngz/while/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/osutmzfngz/while/dsycfvoega/add
=sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOpHsequential_osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp
.sequential/osutmzfngz/while/dsycfvoega/BiasAddBiasAdd.sequential/osutmzfngz/while/dsycfvoega/add:z:0Esequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/osutmzfngz/while/dsycfvoega/BiasAdd²
6sequential/osutmzfngz/while/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/osutmzfngz/while/dsycfvoega/split/split_dimÛ
,sequential/osutmzfngz/while/dsycfvoega/splitSplit?sequential/osutmzfngz/while/dsycfvoega/split/split_dim:output:07sequential/osutmzfngz/while/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/osutmzfngz/while/dsycfvoega/splitë
5sequential/osutmzfngz/while/dsycfvoega/ReadVariableOpReadVariableOp@sequential_osutmzfngz_while_dsycfvoega_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/osutmzfngz/while/dsycfvoega/ReadVariableOpû
*sequential/osutmzfngz/while/dsycfvoega/mulMul=sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp:value:0)sequential_osutmzfngz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/osutmzfngz/while/dsycfvoega/mulþ
,sequential/osutmzfngz/while/dsycfvoega/add_1AddV25sequential/osutmzfngz/while/dsycfvoega/split:output:0.sequential/osutmzfngz/while/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/osutmzfngz/while/dsycfvoega/add_1Ï
.sequential/osutmzfngz/while/dsycfvoega/SigmoidSigmoid0sequential/osutmzfngz/while/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/osutmzfngz/while/dsycfvoega/Sigmoidñ
7sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_1ReadVariableOpBsequential_osutmzfngz_while_dsycfvoega_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_1
,sequential/osutmzfngz/while/dsycfvoega/mul_1Mul?sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_1:value:0)sequential_osutmzfngz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/osutmzfngz/while/dsycfvoega/mul_1
,sequential/osutmzfngz/while/dsycfvoega/add_2AddV25sequential/osutmzfngz/while/dsycfvoega/split:output:10sequential/osutmzfngz/while/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/osutmzfngz/while/dsycfvoega/add_2Ó
0sequential/osutmzfngz/while/dsycfvoega/Sigmoid_1Sigmoid0sequential/osutmzfngz/while/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/osutmzfngz/while/dsycfvoega/Sigmoid_1ö
,sequential/osutmzfngz/while/dsycfvoega/mul_2Mul4sequential/osutmzfngz/while/dsycfvoega/Sigmoid_1:y:0)sequential_osutmzfngz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/osutmzfngz/while/dsycfvoega/mul_2Ë
+sequential/osutmzfngz/while/dsycfvoega/TanhTanh5sequential/osutmzfngz/while/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/osutmzfngz/while/dsycfvoega/Tanhú
,sequential/osutmzfngz/while/dsycfvoega/mul_3Mul2sequential/osutmzfngz/while/dsycfvoega/Sigmoid:y:0/sequential/osutmzfngz/while/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/osutmzfngz/while/dsycfvoega/mul_3û
,sequential/osutmzfngz/while/dsycfvoega/add_3AddV20sequential/osutmzfngz/while/dsycfvoega/mul_2:z:00sequential/osutmzfngz/while/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/osutmzfngz/while/dsycfvoega/add_3ñ
7sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2ReadVariableOpBsequential_osutmzfngz_while_dsycfvoega_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2
,sequential/osutmzfngz/while/dsycfvoega/mul_4Mul?sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2:value:00sequential/osutmzfngz/while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/osutmzfngz/while/dsycfvoega/mul_4
,sequential/osutmzfngz/while/dsycfvoega/add_4AddV25sequential/osutmzfngz/while/dsycfvoega/split:output:30sequential/osutmzfngz/while/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/osutmzfngz/while/dsycfvoega/add_4Ó
0sequential/osutmzfngz/while/dsycfvoega/Sigmoid_2Sigmoid0sequential/osutmzfngz/while/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/osutmzfngz/while/dsycfvoega/Sigmoid_2Ê
-sequential/osutmzfngz/while/dsycfvoega/Tanh_1Tanh0sequential/osutmzfngz/while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/osutmzfngz/while/dsycfvoega/Tanh_1þ
,sequential/osutmzfngz/while/dsycfvoega/mul_5Mul4sequential/osutmzfngz/while/dsycfvoega/Sigmoid_2:y:01sequential/osutmzfngz/while/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/osutmzfngz/while/dsycfvoega/mul_5Ì
@sequential/osutmzfngz/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_osutmzfngz_while_placeholder_1'sequential_osutmzfngz_while_placeholder0sequential/osutmzfngz/while/dsycfvoega/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/osutmzfngz/while/TensorArrayV2Write/TensorListSetItem
!sequential/osutmzfngz/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/osutmzfngz/while/add/yÁ
sequential/osutmzfngz/while/addAddV2'sequential_osutmzfngz_while_placeholder*sequential/osutmzfngz/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/osutmzfngz/while/add
#sequential/osutmzfngz/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/osutmzfngz/while/add_1/yä
!sequential/osutmzfngz/while/add_1AddV2Dsequential_osutmzfngz_while_sequential_osutmzfngz_while_loop_counter,sequential/osutmzfngz/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/osutmzfngz/while/add_1
$sequential/osutmzfngz/while/IdentityIdentity%sequential/osutmzfngz/while/add_1:z:0>^sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp=^sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp?^sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp6^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp8^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_18^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/osutmzfngz/while/Identityµ
&sequential/osutmzfngz/while/Identity_1IdentityJsequential_osutmzfngz_while_sequential_osutmzfngz_while_maximum_iterations>^sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp=^sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp?^sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp6^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp8^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_18^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/osutmzfngz/while/Identity_1
&sequential/osutmzfngz/while/Identity_2Identity#sequential/osutmzfngz/while/add:z:0>^sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp=^sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp?^sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp6^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp8^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_18^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/osutmzfngz/while/Identity_2»
&sequential/osutmzfngz/while/Identity_3IdentityPsequential/osutmzfngz/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp=^sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp?^sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp6^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp8^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_18^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/osutmzfngz/while/Identity_3¬
&sequential/osutmzfngz/while/Identity_4Identity0sequential/osutmzfngz/while/dsycfvoega/mul_5:z:0>^sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp=^sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp?^sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp6^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp8^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_18^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/while/Identity_4¬
&sequential/osutmzfngz/while/Identity_5Identity0sequential/osutmzfngz/while/dsycfvoega/add_3:z:0>^sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp=^sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp?^sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp6^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp8^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_18^sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/osutmzfngz/while/Identity_5"
Fsequential_osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resourceHsequential_osutmzfngz_while_dsycfvoega_biasadd_readvariableop_resource_0"
Gsequential_osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resourceIsequential_osutmzfngz_while_dsycfvoega_matmul_1_readvariableop_resource_0"
Esequential_osutmzfngz_while_dsycfvoega_matmul_readvariableop_resourceGsequential_osutmzfngz_while_dsycfvoega_matmul_readvariableop_resource_0"
@sequential_osutmzfngz_while_dsycfvoega_readvariableop_1_resourceBsequential_osutmzfngz_while_dsycfvoega_readvariableop_1_resource_0"
@sequential_osutmzfngz_while_dsycfvoega_readvariableop_2_resourceBsequential_osutmzfngz_while_dsycfvoega_readvariableop_2_resource_0"
>sequential_osutmzfngz_while_dsycfvoega_readvariableop_resource@sequential_osutmzfngz_while_dsycfvoega_readvariableop_resource_0"U
$sequential_osutmzfngz_while_identity-sequential/osutmzfngz/while/Identity:output:0"Y
&sequential_osutmzfngz_while_identity_1/sequential/osutmzfngz/while/Identity_1:output:0"Y
&sequential_osutmzfngz_while_identity_2/sequential/osutmzfngz/while/Identity_2:output:0"Y
&sequential_osutmzfngz_while_identity_3/sequential/osutmzfngz/while/Identity_3:output:0"Y
&sequential_osutmzfngz_while_identity_4/sequential/osutmzfngz/while/Identity_4:output:0"Y
&sequential_osutmzfngz_while_identity_5/sequential/osutmzfngz/while/Identity_5:output:0"
Asequential_osutmzfngz_while_sequential_osutmzfngz_strided_slice_1Csequential_osutmzfngz_while_sequential_osutmzfngz_strided_slice_1_0"
}sequential_osutmzfngz_while_tensorarrayv2read_tensorlistgetitem_sequential_osutmzfngz_tensorarrayunstack_tensorlistfromtensorsequential_osutmzfngz_while_tensorarrayv2read_tensorlistgetitem_sequential_osutmzfngz_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp=sequential/osutmzfngz/while/dsycfvoega/BiasAdd/ReadVariableOp2|
<sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp<sequential/osutmzfngz/while/dsycfvoega/MatMul/ReadVariableOp2
>sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp>sequential/osutmzfngz/while/dsycfvoega/MatMul_1/ReadVariableOp2n
5sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp5sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp2r
7sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_17sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_12r
7sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_27sequential/osutmzfngz/while/dsycfvoega/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_658156
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_658156___redundant_placeholder04
0while_while_cond_658156___redundant_placeholder14
0while_while_cond_658156___redundant_placeholder24
0while_while_cond_658156___redundant_placeholder34
0while_while_cond_658156___redundant_placeholder44
0while_while_cond_658156___redundant_placeholder54
0while_while_cond_658156___redundant_placeholder6
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
while_cond_656078
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_656078___redundant_placeholder04
0while_while_cond_656078___redundant_placeholder14
0while_while_cond_656078___redundant_placeholder24
0while_while_cond_656078___redundant_placeholder34
0while_while_cond_656078___redundant_placeholder44
0while_while_cond_656078___redundant_placeholder54
0while_while_cond_656078___redundant_placeholder6
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


owshcilvwl_while_cond_6570742
.owshcilvwl_while_owshcilvwl_while_loop_counter8
4owshcilvwl_while_owshcilvwl_while_maximum_iterations 
owshcilvwl_while_placeholder"
owshcilvwl_while_placeholder_1"
owshcilvwl_while_placeholder_2"
owshcilvwl_while_placeholder_34
0owshcilvwl_while_less_owshcilvwl_strided_slice_1J
Fowshcilvwl_while_owshcilvwl_while_cond_657074___redundant_placeholder0J
Fowshcilvwl_while_owshcilvwl_while_cond_657074___redundant_placeholder1J
Fowshcilvwl_while_owshcilvwl_while_cond_657074___redundant_placeholder2J
Fowshcilvwl_while_owshcilvwl_while_cond_657074___redundant_placeholder3J
Fowshcilvwl_while_owshcilvwl_while_cond_657074___redundant_placeholder4J
Fowshcilvwl_while_owshcilvwl_while_cond_657074___redundant_placeholder5J
Fowshcilvwl_while_owshcilvwl_while_cond_657074___redundant_placeholder6
owshcilvwl_while_identity
§
owshcilvwl/while/LessLessowshcilvwl_while_placeholder0owshcilvwl_while_less_owshcilvwl_strided_slice_1*
T0*
_output_shapes
: 2
owshcilvwl/while/Less~
owshcilvwl/while/IdentityIdentityowshcilvwl/while/Less:z:0*
T0
*
_output_shapes
: 2
owshcilvwl/while/Identity"?
owshcilvwl_while_identity"owshcilvwl/while/Identity:output:0*(
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
+__inference_osutmzfngz_layer_call_fn_657684
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_6544022
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
ßY

while_body_658765
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_flzkvrshbq_matmul_readvariableop_resource_0:	 F
3while_flzkvrshbq_matmul_1_readvariableop_resource_0:	 A
2while_flzkvrshbq_biasadd_readvariableop_resource_0:	8
*while_flzkvrshbq_readvariableop_resource_0: :
,while_flzkvrshbq_readvariableop_1_resource_0: :
,while_flzkvrshbq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_flzkvrshbq_matmul_readvariableop_resource:	 D
1while_flzkvrshbq_matmul_1_readvariableop_resource:	 ?
0while_flzkvrshbq_biasadd_readvariableop_resource:	6
(while_flzkvrshbq_readvariableop_resource: 8
*while_flzkvrshbq_readvariableop_1_resource: 8
*while_flzkvrshbq_readvariableop_2_resource: ¢'while/flzkvrshbq/BiasAdd/ReadVariableOp¢&while/flzkvrshbq/MatMul/ReadVariableOp¢(while/flzkvrshbq/MatMul_1/ReadVariableOp¢while/flzkvrshbq/ReadVariableOp¢!while/flzkvrshbq/ReadVariableOp_1¢!while/flzkvrshbq/ReadVariableOp_2Ã
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
&while/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp1while_flzkvrshbq_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/flzkvrshbq/MatMul/ReadVariableOpÑ
while/flzkvrshbq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMulÉ
(while/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp3while_flzkvrshbq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/flzkvrshbq/MatMul_1/ReadVariableOpº
while/flzkvrshbq/MatMul_1MatMulwhile_placeholder_20while/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMul_1°
while/flzkvrshbq/addAddV2!while/flzkvrshbq/MatMul:product:0#while/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/addÂ
'while/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp2while_flzkvrshbq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/flzkvrshbq/BiasAdd/ReadVariableOp½
while/flzkvrshbq/BiasAddBiasAddwhile/flzkvrshbq/add:z:0/while/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/BiasAdd
 while/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/flzkvrshbq/split/split_dim
while/flzkvrshbq/splitSplit)while/flzkvrshbq/split/split_dim:output:0!while/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/flzkvrshbq/split©
while/flzkvrshbq/ReadVariableOpReadVariableOp*while_flzkvrshbq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/flzkvrshbq/ReadVariableOp£
while/flzkvrshbq/mulMul'while/flzkvrshbq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul¦
while/flzkvrshbq/add_1AddV2while/flzkvrshbq/split:output:0while/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_1
while/flzkvrshbq/SigmoidSigmoidwhile/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid¯
!while/flzkvrshbq/ReadVariableOp_1ReadVariableOp,while_flzkvrshbq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_1©
while/flzkvrshbq/mul_1Mul)while/flzkvrshbq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_1¨
while/flzkvrshbq/add_2AddV2while/flzkvrshbq/split:output:1while/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_2
while/flzkvrshbq/Sigmoid_1Sigmoidwhile/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_1
while/flzkvrshbq/mul_2Mulwhile/flzkvrshbq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_2
while/flzkvrshbq/TanhTanhwhile/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh¢
while/flzkvrshbq/mul_3Mulwhile/flzkvrshbq/Sigmoid:y:0while/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_3£
while/flzkvrshbq/add_3AddV2while/flzkvrshbq/mul_2:z:0while/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_3¯
!while/flzkvrshbq/ReadVariableOp_2ReadVariableOp,while_flzkvrshbq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_2°
while/flzkvrshbq/mul_4Mul)while/flzkvrshbq/ReadVariableOp_2:value:0while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_4¨
while/flzkvrshbq/add_4AddV2while/flzkvrshbq/split:output:3while/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_4
while/flzkvrshbq/Sigmoid_2Sigmoidwhile/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_2
while/flzkvrshbq/Tanh_1Tanhwhile/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh_1¦
while/flzkvrshbq/mul_5Mulwhile/flzkvrshbq/Sigmoid_2:y:0while/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/flzkvrshbq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/flzkvrshbq/mul_5:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/flzkvrshbq/add_3:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_flzkvrshbq_biasadd_readvariableop_resource2while_flzkvrshbq_biasadd_readvariableop_resource_0"h
1while_flzkvrshbq_matmul_1_readvariableop_resource3while_flzkvrshbq_matmul_1_readvariableop_resource_0"d
/while_flzkvrshbq_matmul_readvariableop_resource1while_flzkvrshbq_matmul_readvariableop_resource_0"Z
*while_flzkvrshbq_readvariableop_1_resource,while_flzkvrshbq_readvariableop_1_resource_0"Z
*while_flzkvrshbq_readvariableop_2_resource,while_flzkvrshbq_readvariableop_2_resource_0"V
(while_flzkvrshbq_readvariableop_resource*while_flzkvrshbq_readvariableop_resource_0")
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
'while/flzkvrshbq/BiasAdd/ReadVariableOp'while/flzkvrshbq/BiasAdd/ReadVariableOp2P
&while/flzkvrshbq/MatMul/ReadVariableOp&while/flzkvrshbq/MatMul/ReadVariableOp2T
(while/flzkvrshbq/MatMul_1/ReadVariableOp(while/flzkvrshbq/MatMul_1/ReadVariableOp2B
while/flzkvrshbq/ReadVariableOpwhile/flzkvrshbq/ReadVariableOp2F
!while/flzkvrshbq/ReadVariableOp_1!while/flzkvrshbq/ReadVariableOp_12F
!while/flzkvrshbq/ReadVariableOp_2!while/flzkvrshbq/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_658157
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dsycfvoega_matmul_readvariableop_resource_0:	F
3while_dsycfvoega_matmul_1_readvariableop_resource_0:	 A
2while_dsycfvoega_biasadd_readvariableop_resource_0:	8
*while_dsycfvoega_readvariableop_resource_0: :
,while_dsycfvoega_readvariableop_1_resource_0: :
,while_dsycfvoega_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dsycfvoega_matmul_readvariableop_resource:	D
1while_dsycfvoega_matmul_1_readvariableop_resource:	 ?
0while_dsycfvoega_biasadd_readvariableop_resource:	6
(while_dsycfvoega_readvariableop_resource: 8
*while_dsycfvoega_readvariableop_1_resource: 8
*while_dsycfvoega_readvariableop_2_resource: ¢'while/dsycfvoega/BiasAdd/ReadVariableOp¢&while/dsycfvoega/MatMul/ReadVariableOp¢(while/dsycfvoega/MatMul_1/ReadVariableOp¢while/dsycfvoega/ReadVariableOp¢!while/dsycfvoega/ReadVariableOp_1¢!while/dsycfvoega/ReadVariableOp_2Ã
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
&while/dsycfvoega/MatMul/ReadVariableOpReadVariableOp1while_dsycfvoega_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dsycfvoega/MatMul/ReadVariableOpÑ
while/dsycfvoega/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMulÉ
(while/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp3while_dsycfvoega_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dsycfvoega/MatMul_1/ReadVariableOpº
while/dsycfvoega/MatMul_1MatMulwhile_placeholder_20while/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMul_1°
while/dsycfvoega/addAddV2!while/dsycfvoega/MatMul:product:0#while/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/addÂ
'while/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp2while_dsycfvoega_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dsycfvoega/BiasAdd/ReadVariableOp½
while/dsycfvoega/BiasAddBiasAddwhile/dsycfvoega/add:z:0/while/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/BiasAdd
 while/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dsycfvoega/split/split_dim
while/dsycfvoega/splitSplit)while/dsycfvoega/split/split_dim:output:0!while/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dsycfvoega/split©
while/dsycfvoega/ReadVariableOpReadVariableOp*while_dsycfvoega_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dsycfvoega/ReadVariableOp£
while/dsycfvoega/mulMul'while/dsycfvoega/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul¦
while/dsycfvoega/add_1AddV2while/dsycfvoega/split:output:0while/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_1
while/dsycfvoega/SigmoidSigmoidwhile/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid¯
!while/dsycfvoega/ReadVariableOp_1ReadVariableOp,while_dsycfvoega_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_1©
while/dsycfvoega/mul_1Mul)while/dsycfvoega/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_1¨
while/dsycfvoega/add_2AddV2while/dsycfvoega/split:output:1while/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_2
while/dsycfvoega/Sigmoid_1Sigmoidwhile/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_1
while/dsycfvoega/mul_2Mulwhile/dsycfvoega/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_2
while/dsycfvoega/TanhTanhwhile/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh¢
while/dsycfvoega/mul_3Mulwhile/dsycfvoega/Sigmoid:y:0while/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_3£
while/dsycfvoega/add_3AddV2while/dsycfvoega/mul_2:z:0while/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_3¯
!while/dsycfvoega/ReadVariableOp_2ReadVariableOp,while_dsycfvoega_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_2°
while/dsycfvoega/mul_4Mul)while/dsycfvoega/ReadVariableOp_2:value:0while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_4¨
while/dsycfvoega/add_4AddV2while/dsycfvoega/split:output:3while/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_4
while/dsycfvoega/Sigmoid_2Sigmoidwhile/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_2
while/dsycfvoega/Tanh_1Tanhwhile/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh_1¦
while/dsycfvoega/mul_5Mulwhile/dsycfvoega/Sigmoid_2:y:0while/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dsycfvoega/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dsycfvoega/mul_5:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dsycfvoega/add_3:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dsycfvoega_biasadd_readvariableop_resource2while_dsycfvoega_biasadd_readvariableop_resource_0"h
1while_dsycfvoega_matmul_1_readvariableop_resource3while_dsycfvoega_matmul_1_readvariableop_resource_0"d
/while_dsycfvoega_matmul_readvariableop_resource1while_dsycfvoega_matmul_readvariableop_resource_0"Z
*while_dsycfvoega_readvariableop_1_resource,while_dsycfvoega_readvariableop_1_resource_0"Z
*while_dsycfvoega_readvariableop_2_resource,while_dsycfvoega_readvariableop_2_resource_0"V
(while_dsycfvoega_readvariableop_resource*while_dsycfvoega_readvariableop_resource_0")
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
'while/dsycfvoega/BiasAdd/ReadVariableOp'while/dsycfvoega/BiasAdd/ReadVariableOp2P
&while/dsycfvoega/MatMul/ReadVariableOp&while/dsycfvoega/MatMul/ReadVariableOp2T
(while/dsycfvoega/MatMul_1/ReadVariableOp(while/dsycfvoega/MatMul_1/ReadVariableOp2B
while/dsycfvoega/ReadVariableOpwhile/dsycfvoega/ReadVariableOp2F
!while/dsycfvoega/ReadVariableOp_1!while/dsycfvoega/ReadVariableOp_12F
!while/dsycfvoega/ReadVariableOp_2!while/dsycfvoega/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_658866
inputs_0<
)flzkvrshbq_matmul_readvariableop_resource:	 >
+flzkvrshbq_matmul_1_readvariableop_resource:	 9
*flzkvrshbq_biasadd_readvariableop_resource:	0
"flzkvrshbq_readvariableop_resource: 2
$flzkvrshbq_readvariableop_1_resource: 2
$flzkvrshbq_readvariableop_2_resource: 
identity¢!flzkvrshbq/BiasAdd/ReadVariableOp¢ flzkvrshbq/MatMul/ReadVariableOp¢"flzkvrshbq/MatMul_1/ReadVariableOp¢flzkvrshbq/ReadVariableOp¢flzkvrshbq/ReadVariableOp_1¢flzkvrshbq/ReadVariableOp_2¢whileF
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
 flzkvrshbq/MatMul/ReadVariableOpReadVariableOp)flzkvrshbq_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 flzkvrshbq/MatMul/ReadVariableOp§
flzkvrshbq/MatMulMatMulstrided_slice_2:output:0(flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMulµ
"flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp+flzkvrshbq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"flzkvrshbq/MatMul_1/ReadVariableOp£
flzkvrshbq/MatMul_1MatMulzeros:output:0*flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMul_1
flzkvrshbq/addAddV2flzkvrshbq/MatMul:product:0flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/add®
!flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp*flzkvrshbq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!flzkvrshbq/BiasAdd/ReadVariableOp¥
flzkvrshbq/BiasAddBiasAddflzkvrshbq/add:z:0)flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/BiasAddz
flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
flzkvrshbq/split/split_dimë
flzkvrshbq/splitSplit#flzkvrshbq/split/split_dim:output:0flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
flzkvrshbq/split
flzkvrshbq/ReadVariableOpReadVariableOp"flzkvrshbq_readvariableop_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp
flzkvrshbq/mulMul!flzkvrshbq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul
flzkvrshbq/add_1AddV2flzkvrshbq/split:output:0flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_1{
flzkvrshbq/SigmoidSigmoidflzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid
flzkvrshbq/ReadVariableOp_1ReadVariableOp$flzkvrshbq_readvariableop_1_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_1
flzkvrshbq/mul_1Mul#flzkvrshbq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_1
flzkvrshbq/add_2AddV2flzkvrshbq/split:output:1flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_2
flzkvrshbq/Sigmoid_1Sigmoidflzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_1
flzkvrshbq/mul_2Mulflzkvrshbq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_2w
flzkvrshbq/TanhTanhflzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh
flzkvrshbq/mul_3Mulflzkvrshbq/Sigmoid:y:0flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_3
flzkvrshbq/add_3AddV2flzkvrshbq/mul_2:z:0flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_3
flzkvrshbq/ReadVariableOp_2ReadVariableOp$flzkvrshbq_readvariableop_2_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_2
flzkvrshbq/mul_4Mul#flzkvrshbq/ReadVariableOp_2:value:0flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_4
flzkvrshbq/add_4AddV2flzkvrshbq/split:output:3flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_4
flzkvrshbq/Sigmoid_2Sigmoidflzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_2v
flzkvrshbq/Tanh_1Tanhflzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh_1
flzkvrshbq/mul_5Mulflzkvrshbq/Sigmoid_2:y:0flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)flzkvrshbq_matmul_readvariableop_resource+flzkvrshbq_matmul_1_readvariableop_resource*flzkvrshbq_biasadd_readvariableop_resource"flzkvrshbq_readvariableop_resource$flzkvrshbq_readvariableop_1_resource$flzkvrshbq_readvariableop_2_resource*
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
while_body_658765*
condR
while_cond_658764*Q
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
IdentityIdentitystrided_slice_3:output:0"^flzkvrshbq/BiasAdd/ReadVariableOp!^flzkvrshbq/MatMul/ReadVariableOp#^flzkvrshbq/MatMul_1/ReadVariableOp^flzkvrshbq/ReadVariableOp^flzkvrshbq/ReadVariableOp_1^flzkvrshbq/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!flzkvrshbq/BiasAdd/ReadVariableOp!flzkvrshbq/BiasAdd/ReadVariableOp2D
 flzkvrshbq/MatMul/ReadVariableOp flzkvrshbq/MatMul/ReadVariableOp2H
"flzkvrshbq/MatMul_1/ReadVariableOp"flzkvrshbq/MatMul_1/ReadVariableOp26
flzkvrshbq/ReadVariableOpflzkvrshbq/ReadVariableOp2:
flzkvrshbq/ReadVariableOp_1flzkvrshbq/ReadVariableOp_12:
flzkvrshbq/ReadVariableOp_2flzkvrshbq/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Ù

+__inference_osutmzfngz_layer_call_fn_657718

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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_6563942
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
ßY

while_body_655804
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_flzkvrshbq_matmul_readvariableop_resource_0:	 F
3while_flzkvrshbq_matmul_1_readvariableop_resource_0:	 A
2while_flzkvrshbq_biasadd_readvariableop_resource_0:	8
*while_flzkvrshbq_readvariableop_resource_0: :
,while_flzkvrshbq_readvariableop_1_resource_0: :
,while_flzkvrshbq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_flzkvrshbq_matmul_readvariableop_resource:	 D
1while_flzkvrshbq_matmul_1_readvariableop_resource:	 ?
0while_flzkvrshbq_biasadd_readvariableop_resource:	6
(while_flzkvrshbq_readvariableop_resource: 8
*while_flzkvrshbq_readvariableop_1_resource: 8
*while_flzkvrshbq_readvariableop_2_resource: ¢'while/flzkvrshbq/BiasAdd/ReadVariableOp¢&while/flzkvrshbq/MatMul/ReadVariableOp¢(while/flzkvrshbq/MatMul_1/ReadVariableOp¢while/flzkvrshbq/ReadVariableOp¢!while/flzkvrshbq/ReadVariableOp_1¢!while/flzkvrshbq/ReadVariableOp_2Ã
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
&while/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp1while_flzkvrshbq_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/flzkvrshbq/MatMul/ReadVariableOpÑ
while/flzkvrshbq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMulÉ
(while/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp3while_flzkvrshbq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/flzkvrshbq/MatMul_1/ReadVariableOpº
while/flzkvrshbq/MatMul_1MatMulwhile_placeholder_20while/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMul_1°
while/flzkvrshbq/addAddV2!while/flzkvrshbq/MatMul:product:0#while/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/addÂ
'while/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp2while_flzkvrshbq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/flzkvrshbq/BiasAdd/ReadVariableOp½
while/flzkvrshbq/BiasAddBiasAddwhile/flzkvrshbq/add:z:0/while/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/BiasAdd
 while/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/flzkvrshbq/split/split_dim
while/flzkvrshbq/splitSplit)while/flzkvrshbq/split/split_dim:output:0!while/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/flzkvrshbq/split©
while/flzkvrshbq/ReadVariableOpReadVariableOp*while_flzkvrshbq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/flzkvrshbq/ReadVariableOp£
while/flzkvrshbq/mulMul'while/flzkvrshbq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul¦
while/flzkvrshbq/add_1AddV2while/flzkvrshbq/split:output:0while/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_1
while/flzkvrshbq/SigmoidSigmoidwhile/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid¯
!while/flzkvrshbq/ReadVariableOp_1ReadVariableOp,while_flzkvrshbq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_1©
while/flzkvrshbq/mul_1Mul)while/flzkvrshbq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_1¨
while/flzkvrshbq/add_2AddV2while/flzkvrshbq/split:output:1while/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_2
while/flzkvrshbq/Sigmoid_1Sigmoidwhile/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_1
while/flzkvrshbq/mul_2Mulwhile/flzkvrshbq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_2
while/flzkvrshbq/TanhTanhwhile/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh¢
while/flzkvrshbq/mul_3Mulwhile/flzkvrshbq/Sigmoid:y:0while/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_3£
while/flzkvrshbq/add_3AddV2while/flzkvrshbq/mul_2:z:0while/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_3¯
!while/flzkvrshbq/ReadVariableOp_2ReadVariableOp,while_flzkvrshbq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_2°
while/flzkvrshbq/mul_4Mul)while/flzkvrshbq/ReadVariableOp_2:value:0while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_4¨
while/flzkvrshbq/add_4AddV2while/flzkvrshbq/split:output:3while/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_4
while/flzkvrshbq/Sigmoid_2Sigmoidwhile/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_2
while/flzkvrshbq/Tanh_1Tanhwhile/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh_1¦
while/flzkvrshbq/mul_5Mulwhile/flzkvrshbq/Sigmoid_2:y:0while/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/flzkvrshbq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/flzkvrshbq/mul_5:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/flzkvrshbq/add_3:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_flzkvrshbq_biasadd_readvariableop_resource2while_flzkvrshbq_biasadd_readvariableop_resource_0"h
1while_flzkvrshbq_matmul_1_readvariableop_resource3while_flzkvrshbq_matmul_1_readvariableop_resource_0"d
/while_flzkvrshbq_matmul_readvariableop_resource1while_flzkvrshbq_matmul_readvariableop_resource_0"Z
*while_flzkvrshbq_readvariableop_1_resource,while_flzkvrshbq_readvariableop_1_resource_0"Z
*while_flzkvrshbq_readvariableop_2_resource,while_flzkvrshbq_readvariableop_2_resource_0"V
(while_flzkvrshbq_readvariableop_resource*while_flzkvrshbq_readvariableop_resource_0")
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
'while/flzkvrshbq/BiasAdd/ReadVariableOp'while/flzkvrshbq/BiasAdd/ReadVariableOp2P
&while/flzkvrshbq/MatMul/ReadVariableOp&while/flzkvrshbq/MatMul/ReadVariableOp2T
(while/flzkvrshbq/MatMul_1/ReadVariableOp(while/flzkvrshbq/MatMul_1/ReadVariableOp2B
while/flzkvrshbq/ReadVariableOpwhile/flzkvrshbq/ReadVariableOp2F
!while/flzkvrshbq/ReadVariableOp_1!while/flzkvrshbq/ReadVariableOp_12F
!while/flzkvrshbq/ReadVariableOp_2!while/flzkvrshbq/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_658258

inputs<
)dsycfvoega_matmul_readvariableop_resource:	>
+dsycfvoega_matmul_1_readvariableop_resource:	 9
*dsycfvoega_biasadd_readvariableop_resource:	0
"dsycfvoega_readvariableop_resource: 2
$dsycfvoega_readvariableop_1_resource: 2
$dsycfvoega_readvariableop_2_resource: 
identity¢!dsycfvoega/BiasAdd/ReadVariableOp¢ dsycfvoega/MatMul/ReadVariableOp¢"dsycfvoega/MatMul_1/ReadVariableOp¢dsycfvoega/ReadVariableOp¢dsycfvoega/ReadVariableOp_1¢dsycfvoega/ReadVariableOp_2¢whileD
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
 dsycfvoega/MatMul/ReadVariableOpReadVariableOp)dsycfvoega_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dsycfvoega/MatMul/ReadVariableOp§
dsycfvoega/MatMulMatMulstrided_slice_2:output:0(dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMulµ
"dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp+dsycfvoega_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dsycfvoega/MatMul_1/ReadVariableOp£
dsycfvoega/MatMul_1MatMulzeros:output:0*dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMul_1
dsycfvoega/addAddV2dsycfvoega/MatMul:product:0dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/add®
!dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp*dsycfvoega_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dsycfvoega/BiasAdd/ReadVariableOp¥
dsycfvoega/BiasAddBiasAdddsycfvoega/add:z:0)dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/BiasAddz
dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dsycfvoega/split/split_dimë
dsycfvoega/splitSplit#dsycfvoega/split/split_dim:output:0dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dsycfvoega/split
dsycfvoega/ReadVariableOpReadVariableOp"dsycfvoega_readvariableop_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp
dsycfvoega/mulMul!dsycfvoega/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul
dsycfvoega/add_1AddV2dsycfvoega/split:output:0dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_1{
dsycfvoega/SigmoidSigmoiddsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid
dsycfvoega/ReadVariableOp_1ReadVariableOp$dsycfvoega_readvariableop_1_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_1
dsycfvoega/mul_1Mul#dsycfvoega/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_1
dsycfvoega/add_2AddV2dsycfvoega/split:output:1dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_2
dsycfvoega/Sigmoid_1Sigmoiddsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_1
dsycfvoega/mul_2Muldsycfvoega/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_2w
dsycfvoega/TanhTanhdsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh
dsycfvoega/mul_3Muldsycfvoega/Sigmoid:y:0dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_3
dsycfvoega/add_3AddV2dsycfvoega/mul_2:z:0dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_3
dsycfvoega/ReadVariableOp_2ReadVariableOp$dsycfvoega_readvariableop_2_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_2
dsycfvoega/mul_4Mul#dsycfvoega/ReadVariableOp_2:value:0dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_4
dsycfvoega/add_4AddV2dsycfvoega/split:output:3dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_4
dsycfvoega/Sigmoid_2Sigmoiddsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_2v
dsycfvoega/Tanh_1Tanhdsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh_1
dsycfvoega/mul_5Muldsycfvoega/Sigmoid_2:y:0dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dsycfvoega_matmul_readvariableop_resource+dsycfvoega_matmul_1_readvariableop_resource*dsycfvoega_biasadd_readvariableop_resource"dsycfvoega_readvariableop_resource$dsycfvoega_readvariableop_1_resource$dsycfvoega_readvariableop_2_resource*
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
while_body_658157*
condR
while_cond_658156*Q
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
IdentityIdentitytranspose_1:y:0"^dsycfvoega/BiasAdd/ReadVariableOp!^dsycfvoega/MatMul/ReadVariableOp#^dsycfvoega/MatMul_1/ReadVariableOp^dsycfvoega/ReadVariableOp^dsycfvoega/ReadVariableOp_1^dsycfvoega/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dsycfvoega/BiasAdd/ReadVariableOp!dsycfvoega/BiasAdd/ReadVariableOp2D
 dsycfvoega/MatMul/ReadVariableOp dsycfvoega/MatMul/ReadVariableOp2H
"dsycfvoega/MatMul_1/ReadVariableOp"dsycfvoega/MatMul_1/ReadVariableOp26
dsycfvoega/ReadVariableOpdsycfvoega/ReadVariableOp2:
dsycfvoega/ReadVariableOp_1dsycfvoega/ReadVariableOp_12:
dsycfvoega/ReadVariableOp_2dsycfvoega/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
¿
+__inference_flzkvrshbq_layer_call_fn_659425

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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_6549842
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
Ùh

F__inference_owshcilvwl_layer_call_and_return_conditional_losses_658686
inputs_0<
)flzkvrshbq_matmul_readvariableop_resource:	 >
+flzkvrshbq_matmul_1_readvariableop_resource:	 9
*flzkvrshbq_biasadd_readvariableop_resource:	0
"flzkvrshbq_readvariableop_resource: 2
$flzkvrshbq_readvariableop_1_resource: 2
$flzkvrshbq_readvariableop_2_resource: 
identity¢!flzkvrshbq/BiasAdd/ReadVariableOp¢ flzkvrshbq/MatMul/ReadVariableOp¢"flzkvrshbq/MatMul_1/ReadVariableOp¢flzkvrshbq/ReadVariableOp¢flzkvrshbq/ReadVariableOp_1¢flzkvrshbq/ReadVariableOp_2¢whileF
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
 flzkvrshbq/MatMul/ReadVariableOpReadVariableOp)flzkvrshbq_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 flzkvrshbq/MatMul/ReadVariableOp§
flzkvrshbq/MatMulMatMulstrided_slice_2:output:0(flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMulµ
"flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp+flzkvrshbq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"flzkvrshbq/MatMul_1/ReadVariableOp£
flzkvrshbq/MatMul_1MatMulzeros:output:0*flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMul_1
flzkvrshbq/addAddV2flzkvrshbq/MatMul:product:0flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/add®
!flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp*flzkvrshbq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!flzkvrshbq/BiasAdd/ReadVariableOp¥
flzkvrshbq/BiasAddBiasAddflzkvrshbq/add:z:0)flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/BiasAddz
flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
flzkvrshbq/split/split_dimë
flzkvrshbq/splitSplit#flzkvrshbq/split/split_dim:output:0flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
flzkvrshbq/split
flzkvrshbq/ReadVariableOpReadVariableOp"flzkvrshbq_readvariableop_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp
flzkvrshbq/mulMul!flzkvrshbq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul
flzkvrshbq/add_1AddV2flzkvrshbq/split:output:0flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_1{
flzkvrshbq/SigmoidSigmoidflzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid
flzkvrshbq/ReadVariableOp_1ReadVariableOp$flzkvrshbq_readvariableop_1_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_1
flzkvrshbq/mul_1Mul#flzkvrshbq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_1
flzkvrshbq/add_2AddV2flzkvrshbq/split:output:1flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_2
flzkvrshbq/Sigmoid_1Sigmoidflzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_1
flzkvrshbq/mul_2Mulflzkvrshbq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_2w
flzkvrshbq/TanhTanhflzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh
flzkvrshbq/mul_3Mulflzkvrshbq/Sigmoid:y:0flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_3
flzkvrshbq/add_3AddV2flzkvrshbq/mul_2:z:0flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_3
flzkvrshbq/ReadVariableOp_2ReadVariableOp$flzkvrshbq_readvariableop_2_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_2
flzkvrshbq/mul_4Mul#flzkvrshbq/ReadVariableOp_2:value:0flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_4
flzkvrshbq/add_4AddV2flzkvrshbq/split:output:3flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_4
flzkvrshbq/Sigmoid_2Sigmoidflzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_2v
flzkvrshbq/Tanh_1Tanhflzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh_1
flzkvrshbq/mul_5Mulflzkvrshbq/Sigmoid_2:y:0flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)flzkvrshbq_matmul_readvariableop_resource+flzkvrshbq_matmul_1_readvariableop_resource*flzkvrshbq_biasadd_readvariableop_resource"flzkvrshbq_readvariableop_resource$flzkvrshbq_readvariableop_1_resource$flzkvrshbq_readvariableop_2_resource*
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
while_body_658585*
condR
while_cond_658584*Q
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
IdentityIdentitystrided_slice_3:output:0"^flzkvrshbq/BiasAdd/ReadVariableOp!^flzkvrshbq/MatMul/ReadVariableOp#^flzkvrshbq/MatMul_1/ReadVariableOp^flzkvrshbq/ReadVariableOp^flzkvrshbq/ReadVariableOp_1^flzkvrshbq/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!flzkvrshbq/BiasAdd/ReadVariableOp!flzkvrshbq/BiasAdd/ReadVariableOp2D
 flzkvrshbq/MatMul/ReadVariableOp flzkvrshbq/MatMul/ReadVariableOp2H
"flzkvrshbq/MatMul_1/ReadVariableOp"flzkvrshbq/MatMul_1/ReadVariableOp26
flzkvrshbq/ReadVariableOpflzkvrshbq/ReadVariableOp2:
flzkvrshbq/ReadVariableOp_1flzkvrshbq/ReadVariableOp_12:
flzkvrshbq/ReadVariableOp_2flzkvrshbq/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0


å
while_cond_657976
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_657976___redundant_placeholder04
0while_while_cond_657976___redundant_placeholder14
0while_while_cond_657976___redundant_placeholder24
0while_while_cond_657976___redundant_placeholder34
0while_while_cond_657976___redundant_placeholder44
0while_while_cond_657976___redundant_placeholder54
0while_while_cond_657976___redundant_placeholder6
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


osutmzfngz_while_cond_6568982
.osutmzfngz_while_osutmzfngz_while_loop_counter8
4osutmzfngz_while_osutmzfngz_while_maximum_iterations 
osutmzfngz_while_placeholder"
osutmzfngz_while_placeholder_1"
osutmzfngz_while_placeholder_2"
osutmzfngz_while_placeholder_34
0osutmzfngz_while_less_osutmzfngz_strided_slice_1J
Fosutmzfngz_while_osutmzfngz_while_cond_656898___redundant_placeholder0J
Fosutmzfngz_while_osutmzfngz_while_cond_656898___redundant_placeholder1J
Fosutmzfngz_while_osutmzfngz_while_cond_656898___redundant_placeholder2J
Fosutmzfngz_while_osutmzfngz_while_cond_656898___redundant_placeholder3J
Fosutmzfngz_while_osutmzfngz_while_cond_656898___redundant_placeholder4J
Fosutmzfngz_while_osutmzfngz_while_cond_656898___redundant_placeholder5J
Fosutmzfngz_while_osutmzfngz_while_cond_656898___redundant_placeholder6
osutmzfngz_while_identity
§
osutmzfngz/while/LessLessosutmzfngz_while_placeholder0osutmzfngz_while_less_osutmzfngz_strided_slice_1*
T0*
_output_shapes
: 2
osutmzfngz/while/Less~
osutmzfngz/while/IdentityIdentityosutmzfngz/while/Less:z:0*
T0
*
_output_shapes
: 2
osutmzfngz/while/Identity"?
osutmzfngz_while_identity"osutmzfngz/while/Identity:output:0*(
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
while_body_656079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_flzkvrshbq_matmul_readvariableop_resource_0:	 F
3while_flzkvrshbq_matmul_1_readvariableop_resource_0:	 A
2while_flzkvrshbq_biasadd_readvariableop_resource_0:	8
*while_flzkvrshbq_readvariableop_resource_0: :
,while_flzkvrshbq_readvariableop_1_resource_0: :
,while_flzkvrshbq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_flzkvrshbq_matmul_readvariableop_resource:	 D
1while_flzkvrshbq_matmul_1_readvariableop_resource:	 ?
0while_flzkvrshbq_biasadd_readvariableop_resource:	6
(while_flzkvrshbq_readvariableop_resource: 8
*while_flzkvrshbq_readvariableop_1_resource: 8
*while_flzkvrshbq_readvariableop_2_resource: ¢'while/flzkvrshbq/BiasAdd/ReadVariableOp¢&while/flzkvrshbq/MatMul/ReadVariableOp¢(while/flzkvrshbq/MatMul_1/ReadVariableOp¢while/flzkvrshbq/ReadVariableOp¢!while/flzkvrshbq/ReadVariableOp_1¢!while/flzkvrshbq/ReadVariableOp_2Ã
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
&while/flzkvrshbq/MatMul/ReadVariableOpReadVariableOp1while_flzkvrshbq_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/flzkvrshbq/MatMul/ReadVariableOpÑ
while/flzkvrshbq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMulÉ
(while/flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp3while_flzkvrshbq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/flzkvrshbq/MatMul_1/ReadVariableOpº
while/flzkvrshbq/MatMul_1MatMulwhile_placeholder_20while/flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/MatMul_1°
while/flzkvrshbq/addAddV2!while/flzkvrshbq/MatMul:product:0#while/flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/addÂ
'while/flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp2while_flzkvrshbq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/flzkvrshbq/BiasAdd/ReadVariableOp½
while/flzkvrshbq/BiasAddBiasAddwhile/flzkvrshbq/add:z:0/while/flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/flzkvrshbq/BiasAdd
 while/flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/flzkvrshbq/split/split_dim
while/flzkvrshbq/splitSplit)while/flzkvrshbq/split/split_dim:output:0!while/flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/flzkvrshbq/split©
while/flzkvrshbq/ReadVariableOpReadVariableOp*while_flzkvrshbq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/flzkvrshbq/ReadVariableOp£
while/flzkvrshbq/mulMul'while/flzkvrshbq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul¦
while/flzkvrshbq/add_1AddV2while/flzkvrshbq/split:output:0while/flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_1
while/flzkvrshbq/SigmoidSigmoidwhile/flzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid¯
!while/flzkvrshbq/ReadVariableOp_1ReadVariableOp,while_flzkvrshbq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_1©
while/flzkvrshbq/mul_1Mul)while/flzkvrshbq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_1¨
while/flzkvrshbq/add_2AddV2while/flzkvrshbq/split:output:1while/flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_2
while/flzkvrshbq/Sigmoid_1Sigmoidwhile/flzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_1
while/flzkvrshbq/mul_2Mulwhile/flzkvrshbq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_2
while/flzkvrshbq/TanhTanhwhile/flzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh¢
while/flzkvrshbq/mul_3Mulwhile/flzkvrshbq/Sigmoid:y:0while/flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_3£
while/flzkvrshbq/add_3AddV2while/flzkvrshbq/mul_2:z:0while/flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_3¯
!while/flzkvrshbq/ReadVariableOp_2ReadVariableOp,while_flzkvrshbq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/flzkvrshbq/ReadVariableOp_2°
while/flzkvrshbq/mul_4Mul)while/flzkvrshbq/ReadVariableOp_2:value:0while/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_4¨
while/flzkvrshbq/add_4AddV2while/flzkvrshbq/split:output:3while/flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/add_4
while/flzkvrshbq/Sigmoid_2Sigmoidwhile/flzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Sigmoid_2
while/flzkvrshbq/Tanh_1Tanhwhile/flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/Tanh_1¦
while/flzkvrshbq/mul_5Mulwhile/flzkvrshbq/Sigmoid_2:y:0while/flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/flzkvrshbq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/flzkvrshbq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/flzkvrshbq/mul_5:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/flzkvrshbq/add_3:z:0(^while/flzkvrshbq/BiasAdd/ReadVariableOp'^while/flzkvrshbq/MatMul/ReadVariableOp)^while/flzkvrshbq/MatMul_1/ReadVariableOp ^while/flzkvrshbq/ReadVariableOp"^while/flzkvrshbq/ReadVariableOp_1"^while/flzkvrshbq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_flzkvrshbq_biasadd_readvariableop_resource2while_flzkvrshbq_biasadd_readvariableop_resource_0"h
1while_flzkvrshbq_matmul_1_readvariableop_resource3while_flzkvrshbq_matmul_1_readvariableop_resource_0"d
/while_flzkvrshbq_matmul_readvariableop_resource1while_flzkvrshbq_matmul_readvariableop_resource_0"Z
*while_flzkvrshbq_readvariableop_1_resource,while_flzkvrshbq_readvariableop_1_resource_0"Z
*while_flzkvrshbq_readvariableop_2_resource,while_flzkvrshbq_readvariableop_2_resource_0"V
(while_flzkvrshbq_readvariableop_resource*while_flzkvrshbq_readvariableop_resource_0")
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
'while/flzkvrshbq/BiasAdd/ReadVariableOp'while/flzkvrshbq/BiasAdd/ReadVariableOp2P
&while/flzkvrshbq/MatMul/ReadVariableOp&while/flzkvrshbq/MatMul/ReadVariableOp2T
(while/flzkvrshbq/MatMul_1/ReadVariableOp(while/flzkvrshbq/MatMul_1/ReadVariableOp2B
while/flzkvrshbq/ReadVariableOpwhile/flzkvrshbq/ReadVariableOp2F
!while/flzkvrshbq/ReadVariableOp_1!while/flzkvrshbq/ReadVariableOp_12F
!while/flzkvrshbq/ReadVariableOp_2!while/flzkvrshbq/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_655079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_655079___redundant_placeholder04
0while_while_cond_655079___redundant_placeholder14
0while_while_cond_655079___redundant_placeholder24
0while_while_cond_655079___redundant_placeholder34
0while_while_cond_655079___redundant_placeholder44
0while_while_cond_655079___redundant_placeholder54
0while_while_cond_655079___redundant_placeholder6
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
while_cond_658944
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_658944___redundant_placeholder04
0while_while_cond_658944___redundant_placeholder14
0while_while_cond_658944___redundant_placeholder24
0while_while_cond_658944___redundant_placeholder34
0while_while_cond_658944___redundant_placeholder44
0while_while_cond_658944___redundant_placeholder54
0while_while_cond_658944___redundant_placeholder6
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_655905

inputs<
)flzkvrshbq_matmul_readvariableop_resource:	 >
+flzkvrshbq_matmul_1_readvariableop_resource:	 9
*flzkvrshbq_biasadd_readvariableop_resource:	0
"flzkvrshbq_readvariableop_resource: 2
$flzkvrshbq_readvariableop_1_resource: 2
$flzkvrshbq_readvariableop_2_resource: 
identity¢!flzkvrshbq/BiasAdd/ReadVariableOp¢ flzkvrshbq/MatMul/ReadVariableOp¢"flzkvrshbq/MatMul_1/ReadVariableOp¢flzkvrshbq/ReadVariableOp¢flzkvrshbq/ReadVariableOp_1¢flzkvrshbq/ReadVariableOp_2¢whileD
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
 flzkvrshbq/MatMul/ReadVariableOpReadVariableOp)flzkvrshbq_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 flzkvrshbq/MatMul/ReadVariableOp§
flzkvrshbq/MatMulMatMulstrided_slice_2:output:0(flzkvrshbq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMulµ
"flzkvrshbq/MatMul_1/ReadVariableOpReadVariableOp+flzkvrshbq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"flzkvrshbq/MatMul_1/ReadVariableOp£
flzkvrshbq/MatMul_1MatMulzeros:output:0*flzkvrshbq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/MatMul_1
flzkvrshbq/addAddV2flzkvrshbq/MatMul:product:0flzkvrshbq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/add®
!flzkvrshbq/BiasAdd/ReadVariableOpReadVariableOp*flzkvrshbq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!flzkvrshbq/BiasAdd/ReadVariableOp¥
flzkvrshbq/BiasAddBiasAddflzkvrshbq/add:z:0)flzkvrshbq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flzkvrshbq/BiasAddz
flzkvrshbq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
flzkvrshbq/split/split_dimë
flzkvrshbq/splitSplit#flzkvrshbq/split/split_dim:output:0flzkvrshbq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
flzkvrshbq/split
flzkvrshbq/ReadVariableOpReadVariableOp"flzkvrshbq_readvariableop_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp
flzkvrshbq/mulMul!flzkvrshbq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul
flzkvrshbq/add_1AddV2flzkvrshbq/split:output:0flzkvrshbq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_1{
flzkvrshbq/SigmoidSigmoidflzkvrshbq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid
flzkvrshbq/ReadVariableOp_1ReadVariableOp$flzkvrshbq_readvariableop_1_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_1
flzkvrshbq/mul_1Mul#flzkvrshbq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_1
flzkvrshbq/add_2AddV2flzkvrshbq/split:output:1flzkvrshbq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_2
flzkvrshbq/Sigmoid_1Sigmoidflzkvrshbq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_1
flzkvrshbq/mul_2Mulflzkvrshbq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_2w
flzkvrshbq/TanhTanhflzkvrshbq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh
flzkvrshbq/mul_3Mulflzkvrshbq/Sigmoid:y:0flzkvrshbq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_3
flzkvrshbq/add_3AddV2flzkvrshbq/mul_2:z:0flzkvrshbq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_3
flzkvrshbq/ReadVariableOp_2ReadVariableOp$flzkvrshbq_readvariableop_2_resource*
_output_shapes
: *
dtype02
flzkvrshbq/ReadVariableOp_2
flzkvrshbq/mul_4Mul#flzkvrshbq/ReadVariableOp_2:value:0flzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_4
flzkvrshbq/add_4AddV2flzkvrshbq/split:output:3flzkvrshbq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/add_4
flzkvrshbq/Sigmoid_2Sigmoidflzkvrshbq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Sigmoid_2v
flzkvrshbq/Tanh_1Tanhflzkvrshbq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/Tanh_1
flzkvrshbq/mul_5Mulflzkvrshbq/Sigmoid_2:y:0flzkvrshbq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flzkvrshbq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)flzkvrshbq_matmul_readvariableop_resource+flzkvrshbq_matmul_1_readvariableop_resource*flzkvrshbq_biasadd_readvariableop_resource"flzkvrshbq_readvariableop_resource$flzkvrshbq_readvariableop_1_resource$flzkvrshbq_readvariableop_2_resource*
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
while_body_655804*
condR
while_cond_655803*Q
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
IdentityIdentitystrided_slice_3:output:0"^flzkvrshbq/BiasAdd/ReadVariableOp!^flzkvrshbq/MatMul/ReadVariableOp#^flzkvrshbq/MatMul_1/ReadVariableOp^flzkvrshbq/ReadVariableOp^flzkvrshbq/ReadVariableOp_1^flzkvrshbq/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!flzkvrshbq/BiasAdd/ReadVariableOp!flzkvrshbq/BiasAdd/ReadVariableOp2D
 flzkvrshbq/MatMul/ReadVariableOp flzkvrshbq/MatMul/ReadVariableOp2H
"flzkvrshbq/MatMul_1/ReadVariableOp"flzkvrshbq/MatMul_1/ReadVariableOp26
flzkvrshbq/ReadVariableOpflzkvrshbq/ReadVariableOp2:
flzkvrshbq/ReadVariableOp_1flzkvrshbq/ReadVariableOp_12:
flzkvrshbq/ReadVariableOp_2flzkvrshbq/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¸'
´
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_659335

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
+__inference_oaettnoaty_layer_call_fn_659235

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
F__inference_oaettnoaty_layer_call_and_return_conditional_losses_6559292
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
Æ

+__inference_bpstkcuudk_layer_call_fn_657595

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
F__inference_bpstkcuudk_layer_call_and_return_conditional_losses_6555122
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
while_cond_654816
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_654816___redundant_placeholder04
0while_while_cond_654816___redundant_placeholder14
0while_while_cond_654816___redundant_placeholder24
0while_while_cond_654816___redundant_placeholder34
0while_while_cond_654816___redundant_placeholder44
0while_while_cond_654816___redundant_placeholder54
0while_while_cond_654816___redundant_placeholder6
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
while_body_657797
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dsycfvoega_matmul_readvariableop_resource_0:	F
3while_dsycfvoega_matmul_1_readvariableop_resource_0:	 A
2while_dsycfvoega_biasadd_readvariableop_resource_0:	8
*while_dsycfvoega_readvariableop_resource_0: :
,while_dsycfvoega_readvariableop_1_resource_0: :
,while_dsycfvoega_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dsycfvoega_matmul_readvariableop_resource:	D
1while_dsycfvoega_matmul_1_readvariableop_resource:	 ?
0while_dsycfvoega_biasadd_readvariableop_resource:	6
(while_dsycfvoega_readvariableop_resource: 8
*while_dsycfvoega_readvariableop_1_resource: 8
*while_dsycfvoega_readvariableop_2_resource: ¢'while/dsycfvoega/BiasAdd/ReadVariableOp¢&while/dsycfvoega/MatMul/ReadVariableOp¢(while/dsycfvoega/MatMul_1/ReadVariableOp¢while/dsycfvoega/ReadVariableOp¢!while/dsycfvoega/ReadVariableOp_1¢!while/dsycfvoega/ReadVariableOp_2Ã
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
&while/dsycfvoega/MatMul/ReadVariableOpReadVariableOp1while_dsycfvoega_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dsycfvoega/MatMul/ReadVariableOpÑ
while/dsycfvoega/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMulÉ
(while/dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp3while_dsycfvoega_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dsycfvoega/MatMul_1/ReadVariableOpº
while/dsycfvoega/MatMul_1MatMulwhile_placeholder_20while/dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/MatMul_1°
while/dsycfvoega/addAddV2!while/dsycfvoega/MatMul:product:0#while/dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/addÂ
'while/dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp2while_dsycfvoega_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dsycfvoega/BiasAdd/ReadVariableOp½
while/dsycfvoega/BiasAddBiasAddwhile/dsycfvoega/add:z:0/while/dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dsycfvoega/BiasAdd
 while/dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dsycfvoega/split/split_dim
while/dsycfvoega/splitSplit)while/dsycfvoega/split/split_dim:output:0!while/dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dsycfvoega/split©
while/dsycfvoega/ReadVariableOpReadVariableOp*while_dsycfvoega_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dsycfvoega/ReadVariableOp£
while/dsycfvoega/mulMul'while/dsycfvoega/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul¦
while/dsycfvoega/add_1AddV2while/dsycfvoega/split:output:0while/dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_1
while/dsycfvoega/SigmoidSigmoidwhile/dsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid¯
!while/dsycfvoega/ReadVariableOp_1ReadVariableOp,while_dsycfvoega_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_1©
while/dsycfvoega/mul_1Mul)while/dsycfvoega/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_1¨
while/dsycfvoega/add_2AddV2while/dsycfvoega/split:output:1while/dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_2
while/dsycfvoega/Sigmoid_1Sigmoidwhile/dsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_1
while/dsycfvoega/mul_2Mulwhile/dsycfvoega/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_2
while/dsycfvoega/TanhTanhwhile/dsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh¢
while/dsycfvoega/mul_3Mulwhile/dsycfvoega/Sigmoid:y:0while/dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_3£
while/dsycfvoega/add_3AddV2while/dsycfvoega/mul_2:z:0while/dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_3¯
!while/dsycfvoega/ReadVariableOp_2ReadVariableOp,while_dsycfvoega_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dsycfvoega/ReadVariableOp_2°
while/dsycfvoega/mul_4Mul)while/dsycfvoega/ReadVariableOp_2:value:0while/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_4¨
while/dsycfvoega/add_4AddV2while/dsycfvoega/split:output:3while/dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/add_4
while/dsycfvoega/Sigmoid_2Sigmoidwhile/dsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Sigmoid_2
while/dsycfvoega/Tanh_1Tanhwhile/dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/Tanh_1¦
while/dsycfvoega/mul_5Mulwhile/dsycfvoega/Sigmoid_2:y:0while/dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dsycfvoega/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dsycfvoega/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dsycfvoega/mul_5:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dsycfvoega/add_3:z:0(^while/dsycfvoega/BiasAdd/ReadVariableOp'^while/dsycfvoega/MatMul/ReadVariableOp)^while/dsycfvoega/MatMul_1/ReadVariableOp ^while/dsycfvoega/ReadVariableOp"^while/dsycfvoega/ReadVariableOp_1"^while/dsycfvoega/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dsycfvoega_biasadd_readvariableop_resource2while_dsycfvoega_biasadd_readvariableop_resource_0"h
1while_dsycfvoega_matmul_1_readvariableop_resource3while_dsycfvoega_matmul_1_readvariableop_resource_0"d
/while_dsycfvoega_matmul_readvariableop_resource1while_dsycfvoega_matmul_readvariableop_resource_0"Z
*while_dsycfvoega_readvariableop_1_resource,while_dsycfvoega_readvariableop_1_resource_0"Z
*while_dsycfvoega_readvariableop_2_resource,while_dsycfvoega_readvariableop_2_resource_0"V
(while_dsycfvoega_readvariableop_resource*while_dsycfvoega_readvariableop_resource_0")
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
'while/dsycfvoega/BiasAdd/ReadVariableOp'while/dsycfvoega/BiasAdd/ReadVariableOp2P
&while/dsycfvoega/MatMul/ReadVariableOp&while/dsycfvoega/MatMul/ReadVariableOp2T
(while/dsycfvoega/MatMul_1/ReadVariableOp(while/dsycfvoega/MatMul_1/ReadVariableOp2B
while/dsycfvoega/ReadVariableOpwhile/dsycfvoega/ReadVariableOp2F
!while/dsycfvoega/ReadVariableOp_1!while/dsycfvoega/ReadVariableOp_12F
!while/dsycfvoega/ReadVariableOp_2!while/dsycfvoega/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
+__inference_flzkvrshbq_layer_call_fn_659402

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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_6547972
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

¡	
'sequential_osutmzfngz_while_cond_653668H
Dsequential_osutmzfngz_while_sequential_osutmzfngz_while_loop_counterN
Jsequential_osutmzfngz_while_sequential_osutmzfngz_while_maximum_iterations+
'sequential_osutmzfngz_while_placeholder-
)sequential_osutmzfngz_while_placeholder_1-
)sequential_osutmzfngz_while_placeholder_2-
)sequential_osutmzfngz_while_placeholder_3J
Fsequential_osutmzfngz_while_less_sequential_osutmzfngz_strided_slice_1`
\sequential_osutmzfngz_while_sequential_osutmzfngz_while_cond_653668___redundant_placeholder0`
\sequential_osutmzfngz_while_sequential_osutmzfngz_while_cond_653668___redundant_placeholder1`
\sequential_osutmzfngz_while_sequential_osutmzfngz_while_cond_653668___redundant_placeholder2`
\sequential_osutmzfngz_while_sequential_osutmzfngz_while_cond_653668___redundant_placeholder3`
\sequential_osutmzfngz_while_sequential_osutmzfngz_while_cond_653668___redundant_placeholder4`
\sequential_osutmzfngz_while_sequential_osutmzfngz_while_cond_653668___redundant_placeholder5`
\sequential_osutmzfngz_while_sequential_osutmzfngz_while_cond_653668___redundant_placeholder6(
$sequential_osutmzfngz_while_identity
Þ
 sequential/osutmzfngz/while/LessLess'sequential_osutmzfngz_while_placeholderFsequential_osutmzfngz_while_less_sequential_osutmzfngz_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/osutmzfngz/while/Less
$sequential/osutmzfngz/while/IdentityIdentity$sequential/osutmzfngz/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/osutmzfngz/while/Identity"U
$sequential_osutmzfngz_while_identity-sequential/osutmzfngz/while/Identity:output:0*(
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_656394

inputs<
)dsycfvoega_matmul_readvariableop_resource:	>
+dsycfvoega_matmul_1_readvariableop_resource:	 9
*dsycfvoega_biasadd_readvariableop_resource:	0
"dsycfvoega_readvariableop_resource: 2
$dsycfvoega_readvariableop_1_resource: 2
$dsycfvoega_readvariableop_2_resource: 
identity¢!dsycfvoega/BiasAdd/ReadVariableOp¢ dsycfvoega/MatMul/ReadVariableOp¢"dsycfvoega/MatMul_1/ReadVariableOp¢dsycfvoega/ReadVariableOp¢dsycfvoega/ReadVariableOp_1¢dsycfvoega/ReadVariableOp_2¢whileD
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
 dsycfvoega/MatMul/ReadVariableOpReadVariableOp)dsycfvoega_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dsycfvoega/MatMul/ReadVariableOp§
dsycfvoega/MatMulMatMulstrided_slice_2:output:0(dsycfvoega/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMulµ
"dsycfvoega/MatMul_1/ReadVariableOpReadVariableOp+dsycfvoega_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dsycfvoega/MatMul_1/ReadVariableOp£
dsycfvoega/MatMul_1MatMulzeros:output:0*dsycfvoega/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/MatMul_1
dsycfvoega/addAddV2dsycfvoega/MatMul:product:0dsycfvoega/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/add®
!dsycfvoega/BiasAdd/ReadVariableOpReadVariableOp*dsycfvoega_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dsycfvoega/BiasAdd/ReadVariableOp¥
dsycfvoega/BiasAddBiasAdddsycfvoega/add:z:0)dsycfvoega/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dsycfvoega/BiasAddz
dsycfvoega/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dsycfvoega/split/split_dimë
dsycfvoega/splitSplit#dsycfvoega/split/split_dim:output:0dsycfvoega/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dsycfvoega/split
dsycfvoega/ReadVariableOpReadVariableOp"dsycfvoega_readvariableop_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp
dsycfvoega/mulMul!dsycfvoega/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul
dsycfvoega/add_1AddV2dsycfvoega/split:output:0dsycfvoega/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_1{
dsycfvoega/SigmoidSigmoiddsycfvoega/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid
dsycfvoega/ReadVariableOp_1ReadVariableOp$dsycfvoega_readvariableop_1_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_1
dsycfvoega/mul_1Mul#dsycfvoega/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_1
dsycfvoega/add_2AddV2dsycfvoega/split:output:1dsycfvoega/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_2
dsycfvoega/Sigmoid_1Sigmoiddsycfvoega/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_1
dsycfvoega/mul_2Muldsycfvoega/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_2w
dsycfvoega/TanhTanhdsycfvoega/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh
dsycfvoega/mul_3Muldsycfvoega/Sigmoid:y:0dsycfvoega/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_3
dsycfvoega/add_3AddV2dsycfvoega/mul_2:z:0dsycfvoega/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_3
dsycfvoega/ReadVariableOp_2ReadVariableOp$dsycfvoega_readvariableop_2_resource*
_output_shapes
: *
dtype02
dsycfvoega/ReadVariableOp_2
dsycfvoega/mul_4Mul#dsycfvoega/ReadVariableOp_2:value:0dsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_4
dsycfvoega/add_4AddV2dsycfvoega/split:output:3dsycfvoega/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/add_4
dsycfvoega/Sigmoid_2Sigmoiddsycfvoega/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Sigmoid_2v
dsycfvoega/Tanh_1Tanhdsycfvoega/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/Tanh_1
dsycfvoega/mul_5Muldsycfvoega/Sigmoid_2:y:0dsycfvoega/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dsycfvoega/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dsycfvoega_matmul_readvariableop_resource+dsycfvoega_matmul_1_readvariableop_resource*dsycfvoega_biasadd_readvariableop_resource"dsycfvoega_readvariableop_resource$dsycfvoega_readvariableop_1_resource$dsycfvoega_readvariableop_2_resource*
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
while_body_656293*
condR
while_cond_656292*Q
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
IdentityIdentitytranspose_1:y:0"^dsycfvoega/BiasAdd/ReadVariableOp!^dsycfvoega/MatMul/ReadVariableOp#^dsycfvoega/MatMul_1/ReadVariableOp^dsycfvoega/ReadVariableOp^dsycfvoega/ReadVariableOp_1^dsycfvoega/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dsycfvoega/BiasAdd/ReadVariableOp!dsycfvoega/BiasAdd/ReadVariableOp2D
 dsycfvoega/MatMul/ReadVariableOp dsycfvoega/MatMul/ReadVariableOp2H
"dsycfvoega/MatMul_1/ReadVariableOp"dsycfvoega/MatMul_1/ReadVariableOp26
dsycfvoega/ReadVariableOpdsycfvoega/ReadVariableOp2:
dsycfvoega/ReadVariableOp_1dsycfvoega/ReadVariableOp_12:
dsycfvoega/ReadVariableOp_2dsycfvoega/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸'
´
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_659379

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
while_cond_654058
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_654058___redundant_placeholder04
0while_while_cond_654058___redundant_placeholder14
0while_while_cond_654058___redundant_placeholder24
0while_while_cond_654058___redundant_placeholder34
0while_while_cond_654058___redundant_placeholder44
0while_while_cond_654058___redundant_placeholder54
0while_while_cond_654058___redundant_placeholder6
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

jfowsgvbzw;
serving_default_jfowsgvbzw:0ÿÿÿÿÿÿÿÿÿ>

oaettnoaty0
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
_tf_keras_sequential£A{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "jfowsgvbzw"}}, {"class_name": "Conv1D", "config": {"name": "bpstkcuudk", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "xlcvyoxoxq", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "osutmzfngz", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "dsycfvoega", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "owshcilvwl", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "flzkvrshbq", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "oaettnoaty", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "jfowsgvbzw"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "jfowsgvbzw"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "bpstkcuudk", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "xlcvyoxoxq", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "osutmzfngz", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "dsycfvoega", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "owshcilvwl", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "flzkvrshbq", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "oaettnoaty", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
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
{"name": "bpstkcuudk", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "bpstkcuudk", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ÿ
_tf_keras_layerå{"name": "xlcvyoxoxq", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "xlcvyoxoxq", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
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
_tf_keras_rnn_layerä{"name": "osutmzfngz", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "osutmzfngz", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "dsycfvoega", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
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
_tf_keras_rnn_layerê{"name": "owshcilvwl", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "owshcilvwl", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "flzkvrshbq", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
__call__
+&call_and_return_all_conditional_losses"²
_tf_keras_layer{"name": "oaettnoaty", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "oaettnoaty", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
':%2bpstkcuudk/kernel
:2bpstkcuudk/bias
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
_tf_keras_layer¼{"name": "dsycfvoega", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "dsycfvoega", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
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
_tf_keras_layerÀ{"name": "flzkvrshbq", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "flzkvrshbq", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
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
#:! 2oaettnoaty/kernel
:2oaettnoaty/bias
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
/:-	2osutmzfngz/dsycfvoega/kernel
9:7	 2&osutmzfngz/dsycfvoega/recurrent_kernel
):'2osutmzfngz/dsycfvoega/bias
?:= 21osutmzfngz/dsycfvoega/input_gate_peephole_weights
@:> 22osutmzfngz/dsycfvoega/forget_gate_peephole_weights
@:> 22osutmzfngz/dsycfvoega/output_gate_peephole_weights
/:-	 2owshcilvwl/flzkvrshbq/kernel
9:7	 2&owshcilvwl/flzkvrshbq/recurrent_kernel
):'2owshcilvwl/flzkvrshbq/bias
?:= 21owshcilvwl/flzkvrshbq/input_gate_peephole_weights
@:> 22owshcilvwl/flzkvrshbq/forget_gate_peephole_weights
@:> 22owshcilvwl/flzkvrshbq/output_gate_peephole_weights
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
1:/2RMSprop/bpstkcuudk/kernel/rms
':%2RMSprop/bpstkcuudk/bias/rms
-:+ 2RMSprop/oaettnoaty/kernel/rms
':%2RMSprop/oaettnoaty/bias/rms
9:7	2(RMSprop/osutmzfngz/dsycfvoega/kernel/rms
C:A	 22RMSprop/osutmzfngz/dsycfvoega/recurrent_kernel/rms
3:12&RMSprop/osutmzfngz/dsycfvoega/bias/rms
I:G 2=RMSprop/osutmzfngz/dsycfvoega/input_gate_peephole_weights/rms
J:H 2>RMSprop/osutmzfngz/dsycfvoega/forget_gate_peephole_weights/rms
J:H 2>RMSprop/osutmzfngz/dsycfvoega/output_gate_peephole_weights/rms
9:7	 2(RMSprop/owshcilvwl/flzkvrshbq/kernel/rms
C:A	 22RMSprop/owshcilvwl/flzkvrshbq/recurrent_kernel/rms
3:12&RMSprop/owshcilvwl/flzkvrshbq/bias/rms
I:G 2=RMSprop/owshcilvwl/flzkvrshbq/input_gate_peephole_weights/rms
J:H 2>RMSprop/owshcilvwl/flzkvrshbq/forget_gate_peephole_weights/rms
J:H 2>RMSprop/owshcilvwl/flzkvrshbq/output_gate_peephole_weights/rms
ú2÷
+__inference_sequential_layer_call_fn_655971
+__inference_sequential_layer_call_fn_656741
+__inference_sequential_layer_call_fn_656778
+__inference_sequential_layer_call_fn_656577À
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
F__inference_sequential_layer_call_and_return_conditional_losses_657182
F__inference_sequential_layer_call_and_return_conditional_losses_657586
F__inference_sequential_layer_call_and_return_conditional_losses_656618
F__inference_sequential_layer_call_and_return_conditional_losses_656659À
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
!__inference__wrapped_model_653952Á
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

jfowsgvbzwÿÿÿÿÿÿÿÿÿ
Õ2Ò
+__inference_bpstkcuudk_layer_call_fn_657595¢
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
F__inference_bpstkcuudk_layer_call_and_return_conditional_losses_657632¢
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
+__inference_xlcvyoxoxq_layer_call_fn_657637¢
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
F__inference_xlcvyoxoxq_layer_call_and_return_conditional_losses_657650¢
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
+__inference_osutmzfngz_layer_call_fn_657667
+__inference_osutmzfngz_layer_call_fn_657684
+__inference_osutmzfngz_layer_call_fn_657701
+__inference_osutmzfngz_layer_call_fn_657718æ
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_657898
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_658078
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_658258
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_658438æ
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
+__inference_owshcilvwl_layer_call_fn_658455
+__inference_owshcilvwl_layer_call_fn_658472
+__inference_owshcilvwl_layer_call_fn_658489
+__inference_owshcilvwl_layer_call_fn_658506æ
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_658686
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_658866
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_659046
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_659226æ
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
+__inference_oaettnoaty_layer_call_fn_659235¢
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
F__inference_oaettnoaty_layer_call_and_return_conditional_losses_659245¢
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
$__inference_signature_wrapper_656704
jfowsgvbzw"
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
+__inference_dsycfvoega_layer_call_fn_659268
+__inference_dsycfvoega_layer_call_fn_659291¾
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
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_659335
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_659379¾
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
+__inference_flzkvrshbq_layer_call_fn_659402
+__inference_flzkvrshbq_layer_call_fn_659425¾
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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_659469
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_659513¾
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
!__inference__wrapped_model_653952-./012345678"#;¢8
1¢.
,)

jfowsgvbzwÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

oaettnoaty$!

oaettnoatyÿÿÿÿÿÿÿÿÿ¶
F__inference_bpstkcuudk_layer_call_and_return_conditional_losses_657632l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_bpstkcuudk_layer_call_fn_657595_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿË
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_659335-./012¢}
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
F__inference_dsycfvoega_layer_call_and_return_conditional_losses_659379-./012¢}
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
+__inference_dsycfvoega_layer_call_fn_659268ð-./012¢}
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
+__inference_dsycfvoega_layer_call_fn_659291ð-./012¢}
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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_659469345678¢}
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
F__inference_flzkvrshbq_layer_call_and_return_conditional_losses_659513345678¢}
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
+__inference_flzkvrshbq_layer_call_fn_659402ð345678¢}
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
+__inference_flzkvrshbq_layer_call_fn_659425ð345678¢}
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
1/1ÿÿÿÿÿÿÿÿÿ ¦
F__inference_oaettnoaty_layer_call_and_return_conditional_losses_659245\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_oaettnoaty_layer_call_fn_659235O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÜ
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_657898-./012S¢P
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_658078-./012S¢P
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_658258x-./012C¢@
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
F__inference_osutmzfngz_layer_call_and_return_conditional_losses_658438x-./012C¢@
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
+__inference_osutmzfngz_layer_call_fn_657667-./012S¢P
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
+__inference_osutmzfngz_layer_call_fn_657684-./012S¢P
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
+__inference_osutmzfngz_layer_call_fn_657701k-./012C¢@
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
+__inference_osutmzfngz_layer_call_fn_657718k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ Ï
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_658686345678S¢P
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_658866345678S¢P
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_659046t345678C¢@
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
F__inference_owshcilvwl_layer_call_and_return_conditional_losses_659226t345678C¢@
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
+__inference_owshcilvwl_layer_call_fn_658455w345678S¢P
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
+__inference_owshcilvwl_layer_call_fn_658472w345678S¢P
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
+__inference_owshcilvwl_layer_call_fn_658489g345678C¢@
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
+__inference_owshcilvwl_layer_call_fn_658506g345678C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ È
F__inference_sequential_layer_call_and_return_conditional_losses_656618~-./012345678"#C¢@
9¢6
,)

jfowsgvbzwÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
F__inference_sequential_layer_call_and_return_conditional_losses_656659~-./012345678"#C¢@
9¢6
,)

jfowsgvbzwÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
F__inference_sequential_layer_call_and_return_conditional_losses_657182z-./012345678"#?¢<
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
F__inference_sequential_layer_call_and_return_conditional_losses_657586z-./012345678"#?¢<
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
+__inference_sequential_layer_call_fn_655971q-./012345678"#C¢@
9¢6
,)

jfowsgvbzwÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
+__inference_sequential_layer_call_fn_656577q-./012345678"#C¢@
9¢6
,)

jfowsgvbzwÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_656741m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_656778m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¿
$__inference_signature_wrapper_656704-./012345678"#I¢F
¢ 
?ª<
:

jfowsgvbzw,)

jfowsgvbzwÿÿÿÿÿÿÿÿÿ"7ª4
2

oaettnoaty$!

oaettnoatyÿÿÿÿÿÿÿÿÿ®
F__inference_xlcvyoxoxq_layer_call_and_return_conditional_losses_657650d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_xlcvyoxoxq_layer_call_fn_657637W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ