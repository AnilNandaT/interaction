??8
?%?%
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
*
Erf
x"T
y"T"
Ttype:
2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
ExtractImagePatches
images"T
patches"T"
ksizes	list(int)(0"
strides	list(int)(0"
rates	list(int)(0"
Ttype:
2	
""
paddingstring:
SAMEVALID
,
Floor
x"T
y"T"
Ttype:
2
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
y
Roll

input"T
shift"Tshift
axis"Taxis
output"T"	
Ttype"
Tshifttype:
2	"
Taxistype:
2	
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??2
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
h
StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
StateVar
a
StateVar/Read/ReadVariableOpReadVariableOpStateVar*
_output_shapes
:*
dtype0	
l

StateVar_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
StateVar_1
e
StateVar_1/Read/ReadVariableOpReadVariableOp
StateVar_1*
_output_shapes
:*
dtype0	
?
patch_merging/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_namepatch_merging/dense_9/kernel
?
0patch_merging/dense_9/kernel/Read/ReadVariableOpReadVariableOppatch_merging/dense_9/kernel* 
_output_shapes
:
??*
dtype0
?
.swin_transformer_1/window_attention_1/VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape
:*?
shared_name0.swin_transformer_1/window_attention_1/Variable
?
Bswin_transformer_1/window_attention_1/Variable/Read/ReadVariableOpReadVariableOp.swin_transformer_1/window_attention_1/Variable*
_output_shapes

:*
dtype0	
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:@*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	?@*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:?*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	@?*
dtype0
?
-swin_transformer_1/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-swin_transformer_1/layer_normalization_3/beta
?
Aswin_transformer_1/layer_normalization_3/beta/Read/ReadVariableOpReadVariableOp-swin_transformer_1/layer_normalization_3/beta*
_output_shapes
:@*
dtype0
?
.swin_transformer_1/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.swin_transformer_1/layer_normalization_3/gamma
?
Bswin_transformer_1/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp.swin_transformer_1/layer_normalization_3/gamma*
_output_shapes
:@*
dtype0
?
2swin_transformer_1/window_attention_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42swin_transformer_1/window_attention_1/dense_6/bias
?
Fswin_transformer_1/window_attention_1/dense_6/bias/Read/ReadVariableOpReadVariableOp2swin_transformer_1/window_attention_1/dense_6/bias*
_output_shapes
:@*
dtype0
?
4swin_transformer_1/window_attention_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*E
shared_name64swin_transformer_1/window_attention_1/dense_6/kernel
?
Hswin_transformer_1/window_attention_1/dense_6/kernel/Read/ReadVariableOpReadVariableOp4swin_transformer_1/window_attention_1/dense_6/kernel*
_output_shapes

:@@*
dtype0
?
2swin_transformer_1/window_attention_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*C
shared_name42swin_transformer_1/window_attention_1/dense_5/bias
?
Fswin_transformer_1/window_attention_1/dense_5/bias/Read/ReadVariableOpReadVariableOp2swin_transformer_1/window_attention_1/dense_5/bias*
_output_shapes	
:?*
dtype0
?
4swin_transformer_1/window_attention_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*E
shared_name64swin_transformer_1/window_attention_1/dense_5/kernel
?
Hswin_transformer_1/window_attention_1/dense_5/kernel/Read/ReadVariableOpReadVariableOp4swin_transformer_1/window_attention_1/dense_5/kernel*
_output_shapes
:	@?*
dtype0
?
,swin_transformer_1/window_attention_1/weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*=
shared_name.,swin_transformer_1/window_attention_1/weight
?
@swin_transformer_1/window_attention_1/weight/Read/ReadVariableOpReadVariableOp,swin_transformer_1/window_attention_1/weight*
_output_shapes

:	*
dtype0
?
-swin_transformer_1/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-swin_transformer_1/layer_normalization_2/beta
?
Aswin_transformer_1/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp-swin_transformer_1/layer_normalization_2/beta*
_output_shapes
:@*
dtype0
?
.swin_transformer_1/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.swin_transformer_1/layer_normalization_2/gamma
?
Bswin_transformer_1/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp.swin_transformer_1/layer_normalization_2/gamma*
_output_shapes
:@*
dtype0
?
*swin_transformer/window_attention/VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape
:*;
shared_name,*swin_transformer/window_attention/Variable
?
>swin_transformer/window_attention/Variable/Read/ReadVariableOpReadVariableOp*swin_transformer/window_attention/Variable*
_output_shapes

:*
dtype0	
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:@*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	?@*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:?*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	@?*
dtype0
?
+swin_transformer/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+swin_transformer/layer_normalization_1/beta
?
?swin_transformer/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp+swin_transformer/layer_normalization_1/beta*
_output_shapes
:@*
dtype0
?
,swin_transformer/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,swin_transformer/layer_normalization_1/gamma
?
@swin_transformer/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp,swin_transformer/layer_normalization_1/gamma*
_output_shapes
:@*
dtype0
?
.swin_transformer/window_attention/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.swin_transformer/window_attention/dense_2/bias
?
Bswin_transformer/window_attention/dense_2/bias/Read/ReadVariableOpReadVariableOp.swin_transformer/window_attention/dense_2/bias*
_output_shapes
:@*
dtype0
?
0swin_transformer/window_attention/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*A
shared_name20swin_transformer/window_attention/dense_2/kernel
?
Dswin_transformer/window_attention/dense_2/kernel/Read/ReadVariableOpReadVariableOp0swin_transformer/window_attention/dense_2/kernel*
_output_shapes

:@@*
dtype0
?
.swin_transformer/window_attention/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.swin_transformer/window_attention/dense_1/bias
?
Bswin_transformer/window_attention/dense_1/bias/Read/ReadVariableOpReadVariableOp.swin_transformer/window_attention/dense_1/bias*
_output_shapes	
:?*
dtype0
?
0swin_transformer/window_attention/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*A
shared_name20swin_transformer/window_attention/dense_1/kernel
?
Dswin_transformer/window_attention/dense_1/kernel/Read/ReadVariableOpReadVariableOp0swin_transformer/window_attention/dense_1/kernel*
_output_shapes
:	@?*
dtype0
?
(swin_transformer/window_attention/weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*9
shared_name*(swin_transformer/window_attention/weight
?
<swin_transformer/window_attention/weight/Read/ReadVariableOpReadVariableOp(swin_transformer/window_attention/weight*
_output_shapes

:	*
dtype0
?
)swin_transformer/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)swin_transformer/layer_normalization/beta
?
=swin_transformer/layer_normalization/beta/Read/ReadVariableOpReadVariableOp)swin_transformer/layer_normalization/beta*
_output_shapes
:@*
dtype0
?
*swin_transformer/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*swin_transformer/layer_normalization/gamma
?
>swin_transformer/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp*swin_transformer/layer_normalization/gamma*
_output_shapes
:@*
dtype0
?
$patch_embedding/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*5
shared_name&$patch_embedding/embedding/embeddings
?
8patch_embedding/embedding/embeddings/Read/ReadVariableOpReadVariableOp$patch_embedding/embedding/embeddings*
_output_shapes
:	?@*
dtype0
?
patch_embedding/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namepatch_embedding/dense/bias
?
.patch_embedding/dense/bias/Read/ReadVariableOpReadVariableOppatch_embedding/dense/bias*
_output_shapes
:@*
dtype0
?
patch_embedding/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_namepatch_embedding/dense/kernel
?
0patch_embedding/dense/kernel/Read/ReadVariableOpReadVariableOppatch_embedding/dense/kernel*
_output_shapes

:@*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:W*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:W*
dtype0
{
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?W* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	?W*
dtype0
?
swin_transformer_1/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameswin_transformer_1/Variable
?
/swin_transformer_1/Variable/Read/ReadVariableOpReadVariableOpswin_transformer_1/Variable*#
_output_shapes
:?*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures
#_self_saveable_object_factories*
'
#_self_saveable_object_factories* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
#_self_saveable_object_factories*
?
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_random_generator
#$_self_saveable_object_factories*
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
#+_self_saveable_object_factories* 
?
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2proj
3	pos_embed
#4_self_saveable_object_factories*
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
	;norm1
<attn
=	drop_path
	>norm2
?mlp
#@_self_saveable_object_factories*
?
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
	Gnorm1
Hattn
I	drop_path
	Jnorm2
Kmlp
L	attn_mask
#M_self_saveable_object_factories*
?
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Tlinear_trans
#U_self_saveable_object_factories*
?
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
#\_self_saveable_object_factories* 
?
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias
#e_self_saveable_object_factories*
?
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11
r12
s13
t14
u15
v16
w17
x18
y19
z20
{21
|22
}23
~24
25
?26
?27
?28
?29
L30
?31
?32
c33
d34*
?
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11
r12
s13
t14
u15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25
?26
?27
?28
?29
c30
d31*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 

?serving_default* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 

?
_generator*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 

?
_generator*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 

f0
g1
h2*

f0
g1
h2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

fkernel
gbias
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
h
embeddings
$?_self_saveable_object_factories*
* 
j
i0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12
v13*
b
i0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	igamma
jbeta
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?qkv
?dropout
	?proj

kweight
 krelative_position_bias_table
vrelative_position_index
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	pgamma
qbeta
$?_self_saveable_object_factories*
?
?layer_with_weights-0
?layer-0
?layer-1
?layer-2
?layer_with_weights-1
?layer-3
?layer-4
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories*
* 
w
w0
x1
y2
z3
{4
|5
}6
~7
8
?9
?10
?11
?12
L13
?14*
f
w0
x1
y2
z3
{4
|5
}6
~7
8
?9
?10
?11
?12*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	wgamma
xbeta
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?qkv
?dropout
	?proj

yweight
 yrelative_position_bias_table
?relative_position_index
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	~gamma
beta
$?_self_saveable_object_factories*
?
?layer_with_weights-0
?layer-0
?layer-1
?layer-2
?layer_with_weights-1
?layer-3
?layer-4
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories*
nh
VARIABLE_VALUEswin_transformer_1/Variable9layer_with_weights-2/attn_mask/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0*

?0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
$?_self_saveable_object_factories*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 

c0
d1*

c0
d1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
\V
VARIABLE_VALUEpatch_embedding/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEpatch_embedding/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$patch_embedding/embedding/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE*swin_transformer/layer_normalization/gamma&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)swin_transformer/layer_normalization/beta&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(swin_transformer/window_attention/weight&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0swin_transformer/window_attention/dense_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.swin_transformer/window_attention/dense_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0swin_transformer/window_attention/dense_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.swin_transformer/window_attention/dense_2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,swin_transformer/layer_normalization_1/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+swin_transformer/layer_normalization_1/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*swin_transformer/window_attention/Variable'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.swin_transformer_1/layer_normalization_2/gamma'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-swin_transformer_1/layer_normalization_2/beta'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,swin_transformer_1/window_attention_1/weight'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4swin_transformer_1/window_attention_1/dense_5/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2swin_transformer_1/window_attention_1/dense_5/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4swin_transformer_1/window_attention_1/dense_6/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2swin_transformer_1/window_attention_1/dense_6/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.swin_transformer_1/layer_normalization_3/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-swin_transformer_1/layer_normalization_3/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_7/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_7/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_8/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_8/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.swin_transformer_1/window_attention_1/Variable'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEpatch_merging/dense_9/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*

v0
L1
?2*
J
0
1
2
3
4
5
6
7
	8

9*

?0
?1
?2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 

?
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 

20
31*
* 
* 
* 
* 
* 

f0
g1*

f0
g1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 

h0*

h0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 

v0*
'
;0
<1
=2
>3
?4*
* 
* 
* 
* 
* 
* 
* 

i0
j1*

i0
j1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
.
k0
l1
m2
n3
o4
v5*
'
k0
l1
m2
n3
o4*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

lkernel
mbias
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

nkernel
obias
$?_self_saveable_object_factories*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 

p0
q1*

p0
q1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

rkernel
sbias
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

tkernel
ubias
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
 
r0
s1
t2
u3*
 
r0
s1
t2
u3*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 

L0
?1*
'
G0
H1
I2
J3
K4*
* 
* 
* 
* 
* 
* 
* 

w0
x1*

w0
x1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
/
y0
z1
{2
|3
}4
?5*
'
y0
z1
{2
|3
}4*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

zkernel
{bias
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

|kernel
}bias
$?_self_saveable_object_factories*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 

~0
1*

~0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
$
?0
?1
?2
?3*
$
?0
?1
?2
?3*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
* 

T0*
* 
* 
* 
* 
* 

?0*

?0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
nh
VARIABLE_VALUE
StateVar_1Jlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEStateVarJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

v0*

?0
?1
?2*
* 
* 
* 

l0
m1*

l0
m1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 

n0
o1*

n0
o1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

r0
s1*

r0
s1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

t0
u1*

t0
u1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
,
?0
?1
?2
?3
?4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0*

?0
?1
?2*
* 
* 
* 

z0
{1*

z0
{1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 

|0
}1*

|0
}1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
,
?0
?1
?2
?3
?4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1patch_embedding/dense/kernelpatch_embedding/dense/bias$patch_embedding/embedding/embeddings*swin_transformer/layer_normalization/gamma)swin_transformer/layer_normalization/beta0swin_transformer/window_attention/dense_1/kernel.swin_transformer/window_attention/dense_1/bias*swin_transformer/window_attention/Variable(swin_transformer/window_attention/weight0swin_transformer/window_attention/dense_2/kernel.swin_transformer/window_attention/dense_2/bias,swin_transformer/layer_normalization_1/gamma+swin_transformer/layer_normalization_1/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/bias.swin_transformer_1/layer_normalization_2/gamma-swin_transformer_1/layer_normalization_2/beta4swin_transformer_1/window_attention_1/dense_5/kernel2swin_transformer_1/window_attention_1/dense_5/bias.swin_transformer_1/window_attention_1/Variable,swin_transformer_1/window_attention_1/weightswin_transformer_1/Variable4swin_transformer_1/window_attention_1/dense_6/kernel2swin_transformer_1/window_attention_1/dense_6/bias.swin_transformer_1/layer_normalization_3/gamma-swin_transformer_1/layer_normalization_3/betadense_7/kerneldense_7/biasdense_8/kerneldense_8/biaspatch_merging/dense_9/kerneldense_10/kerneldense_10/bias*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*E
_read_only_resource_inputs'
%#	
 !"#*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_11131
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/swin_transformer_1/Variable/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp0patch_embedding/dense/kernel/Read/ReadVariableOp.patch_embedding/dense/bias/Read/ReadVariableOp8patch_embedding/embedding/embeddings/Read/ReadVariableOp>swin_transformer/layer_normalization/gamma/Read/ReadVariableOp=swin_transformer/layer_normalization/beta/Read/ReadVariableOp<swin_transformer/window_attention/weight/Read/ReadVariableOpDswin_transformer/window_attention/dense_1/kernel/Read/ReadVariableOpBswin_transformer/window_attention/dense_1/bias/Read/ReadVariableOpDswin_transformer/window_attention/dense_2/kernel/Read/ReadVariableOpBswin_transformer/window_attention/dense_2/bias/Read/ReadVariableOp@swin_transformer/layer_normalization_1/gamma/Read/ReadVariableOp?swin_transformer/layer_normalization_1/beta/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp>swin_transformer/window_attention/Variable/Read/ReadVariableOpBswin_transformer_1/layer_normalization_2/gamma/Read/ReadVariableOpAswin_transformer_1/layer_normalization_2/beta/Read/ReadVariableOp@swin_transformer_1/window_attention_1/weight/Read/ReadVariableOpHswin_transformer_1/window_attention_1/dense_5/kernel/Read/ReadVariableOpFswin_transformer_1/window_attention_1/dense_5/bias/Read/ReadVariableOpHswin_transformer_1/window_attention_1/dense_6/kernel/Read/ReadVariableOpFswin_transformer_1/window_attention_1/dense_6/bias/Read/ReadVariableOpBswin_transformer_1/layer_normalization_3/gamma/Read/ReadVariableOpAswin_transformer_1/layer_normalization_3/beta/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpBswin_transformer_1/window_attention_1/Variable/Read/ReadVariableOp0patch_merging/dense_9/kernel/Read/ReadVariableOpStateVar_1/Read/ReadVariableOpStateVar/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*8
Tin1
/2-				*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_13522
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameswin_transformer_1/Variabledense_10/kerneldense_10/biaspatch_embedding/dense/kernelpatch_embedding/dense/bias$patch_embedding/embedding/embeddings*swin_transformer/layer_normalization/gamma)swin_transformer/layer_normalization/beta(swin_transformer/window_attention/weight0swin_transformer/window_attention/dense_1/kernel.swin_transformer/window_attention/dense_1/bias0swin_transformer/window_attention/dense_2/kernel.swin_transformer/window_attention/dense_2/bias,swin_transformer/layer_normalization_1/gamma+swin_transformer/layer_normalization_1/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*swin_transformer/window_attention/Variable.swin_transformer_1/layer_normalization_2/gamma-swin_transformer_1/layer_normalization_2/beta,swin_transformer_1/window_attention_1/weight4swin_transformer_1/window_attention_1/dense_5/kernel2swin_transformer_1/window_attention_1/dense_5/bias4swin_transformer_1/window_attention_1/dense_6/kernel2swin_transformer_1/window_attention_1/dense_6/bias.swin_transformer_1/layer_normalization_3/gamma-swin_transformer_1/layer_normalization_3/betadense_7/kerneldense_7/biasdense_8/kerneldense_8/bias.swin_transformer_1/window_attention_1/Variablepatch_merging/dense_9/kernel
StateVar_1StateVartotal_2count_2total_1count_1totalcount*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_13661??/
?r
?
cond_true_10345
cond_shape_inputs;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/Assert?$cond/stateful_uniform/RngReadAndSkipK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/sub/yConst*
_output_shapes
: *
dtype0*
value	B :@b
cond/subSubcond/strided_slice:output:0cond/sub/y:output:0*
T0*
_output_shapes
: m
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :@h

cond/sub_1Subcond/strided_slice_1:output:0cond/sub_1/y:output:0*
T0*
_output_shapes
: e
cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:[
cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : _
cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????e
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
cond/stateful_uniform/ProdProd$cond/stateful_uniform/shape:output:0$cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ^
cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :y
cond/stateful_uniform/Cast_1Cast#cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$cond/stateful_uniform/RngReadAndSkipRngReadAndSkip-cond_stateful_uniform_rngreadandskip_resource%cond/stateful_uniform/Cast/x:output:0 cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:s
)cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
cond/stateful_uniform/BitcastBitcast,cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0u
+cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
cond/stateful_uniformStatelessRandomUniformIntV2$cond/stateful_uniform/shape:output:0(cond/stateful_uniform/Bitcast_1:output:0&cond/stateful_uniform/Bitcast:output:0"cond/stateful_uniform/alg:output:0"cond/stateful_uniform/min:output:0"cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0d
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_2StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :U
cond/addAddV2cond/sub:z:0cond/add/y:output:0*
T0*
_output_shapes
: b
cond/modFloorModcond/strided_slice_2:output:0cond/add:z:0*
T0*
_output_shapes
: d
cond/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_3StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_3/stack:output:0%cond/strided_slice_3/stack_1:output:0%cond/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :[

cond/add_1AddV2cond/sub_1:z:0cond/add_1/y:output:0*
T0*
_output_shapes
: f

cond/mod_1FloorModcond/strided_slice_3:output:0cond/add_1:z:0*
T0*
_output_shapes
: `
cond/crop_to_bounding_box/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B :@?
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B :@?
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B :@?
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B :@?
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentitycond_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????@@c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B :@e
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B :@?
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????@@?
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:?????????@@?
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: 2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:5 1
/
_output_shapes
:?????????@@
?
E
)__inference_dropout_4_layer_call_fn_13282

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12468f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_random_flip_layer_call_and_return_conditional_losses_12016

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?5
?
G__inference_patch_merging_layer_call_and_return_conditional_losses_2938
x=
)dense_9_tensordot_readvariableop_resource:
??
identity?? dense_9/Tensordot/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   g
ReshapeReshapexReshape/shape:output:0*
T0*/
_output_shapes
:?????????  @l
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ?
strided_sliceStridedSliceReshape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask	*
end_maskn
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"               p
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                p
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ?
strided_slice_1StridedSliceReshape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask	*
end_maskn
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"               p
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                p
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ?
strided_slice_2StridedSliceReshape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask	*
end_maskn
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"              p
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                p
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ?
strided_slice_3StridedSliceReshape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask	*
end_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0strided_slice_3:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      w
	Reshape_1Reshapeconcat:output:0Reshape_1/shape:output:0*
T0*-
_output_shapes
:????????????
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0`
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Y
dense_9/Tensordot/ShapeShapeReshape_1:output:0*
T0*
_output_shapes
:a
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_9/Tensordot/transpose	TransposeReshape_1:output:0!dense_9/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?a
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????i
NoOpNoOp!^dense_9/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentitydense_9/Tensordot:output:0^NoOp*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????@: 2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:O K
,
_output_shapes
:??????????@

_user_specified_namex
?6
?
@__inference_model_layer_call_and_return_conditional_losses_10105

inputs'
patch_embedding_10016:@#
patch_embedding_10018:@(
patch_embedding_10020:	?@$
swin_transformer_10023:@$
swin_transformer_10025:@)
swin_transformer_10027:	@?%
swin_transformer_10029:	?(
swin_transformer_10031:	(
swin_transformer_10033:	(
swin_transformer_10035:@@$
swin_transformer_10037:@$
swin_transformer_10039:@$
swin_transformer_10041:@)
swin_transformer_10043:	@?%
swin_transformer_10045:	?)
swin_transformer_10047:	?@$
swin_transformer_10049:@&
swin_transformer_1_10052:@&
swin_transformer_1_10054:@+
swin_transformer_1_10056:	@?'
swin_transformer_1_10058:	?*
swin_transformer_1_10060:	*
swin_transformer_1_10062:	/
swin_transformer_1_10064:?*
swin_transformer_1_10066:@@&
swin_transformer_1_10068:@&
swin_transformer_1_10070:@&
swin_transformer_1_10072:@+
swin_transformer_1_10074:	@?'
swin_transformer_1_10076:	?+
swin_transformer_1_10078:	?@&
swin_transformer_1_10080:@'
patch_merging_10083:
??!
dense_10_10099:	?W
dense_10_10101:W
identity?? dense_10/StatefulPartitionedCall?'patch_embedding/StatefulPartitionedCall?%patch_merging/StatefulPartitionedCall?(swin_transformer/StatefulPartitionedCall?*swin_transformer_1/StatefulPartitionedCall?
random_crop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10007?
random_flip/PartitionedCallPartitionedCall$random_crop/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10013?
patch_extract/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9773?
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_10016patch_embedding_10018patch_embedding_10020*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9785?
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_10023swin_transformer_10025swin_transformer_10027swin_transformer_10029swin_transformer_10031swin_transformer_10033swin_transformer_10035swin_transformer_10037swin_transformer_10039swin_transformer_10041swin_transformer_10043swin_transformer_10045swin_transformer_10047swin_transformer_10049*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9825?
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_10052swin_transformer_1_10054swin_transformer_1_10056swin_transformer_1_10058swin_transformer_1_10060swin_transformer_1_10062swin_transformer_1_10064swin_transformer_1_10066swin_transformer_1_10068swin_transformer_1_10070swin_transformer_1_10072swin_transformer_1_10074swin_transformer_1_10076swin_transformer_1_10078swin_transformer_1_10080*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9889?
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_10083*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9927?
(global_average_pooling1d/PartitionedCallPartitionedCall.patch_merging/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_10_10099dense_10_10101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_10098x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????W?
NoOpNoOp!^dense_10/StatefulPartitionedCall(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?9
?
@__inference_model_layer_call_and_return_conditional_losses_11054
input_1
random_crop_10970:	
random_flip_10973:	'
patch_embedding_10977:@#
patch_embedding_10979:@(
patch_embedding_10981:	?@$
swin_transformer_10984:@$
swin_transformer_10986:@)
swin_transformer_10988:	@?%
swin_transformer_10990:	?(
swin_transformer_10992:	(
swin_transformer_10994:	(
swin_transformer_10996:@@$
swin_transformer_10998:@$
swin_transformer_11000:@$
swin_transformer_11002:@)
swin_transformer_11004:	@?%
swin_transformer_11006:	?)
swin_transformer_11008:	?@$
swin_transformer_11010:@&
swin_transformer_1_11013:@&
swin_transformer_1_11015:@+
swin_transformer_1_11017:	@?'
swin_transformer_1_11019:	?*
swin_transformer_1_11021:	*
swin_transformer_1_11023:	/
swin_transformer_1_11025:?*
swin_transformer_1_11027:@@&
swin_transformer_1_11029:@&
swin_transformer_1_11031:@&
swin_transformer_1_11033:@+
swin_transformer_1_11035:	@?'
swin_transformer_1_11037:	?+
swin_transformer_1_11039:	?@&
swin_transformer_1_11041:@'
patch_merging_11044:
??!
dense_10_11048:	?W
dense_10_11050:W
identity?? dense_10/StatefulPartitionedCall?'patch_embedding/StatefulPartitionedCall?%patch_merging/StatefulPartitionedCall?#random_crop/StatefulPartitionedCall?#random_flip/StatefulPartitionedCall?(swin_transformer/StatefulPartitionedCall?*swin_transformer_1/StatefulPartitionedCall?
#random_crop/StatefulPartitionedCallStatefulPartitionedCallinput_1random_crop_10970*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10491?
#random_flip/StatefulPartitionedCallStatefulPartitionedCall,random_crop/StatefulPartitionedCall:output:0random_flip_10973*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10305?
patch_extract/PartitionedCallPartitionedCall,random_flip/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9773?
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_10977patch_embedding_10979patch_embedding_10981*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9785?
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_10984swin_transformer_10986swin_transformer_10988swin_transformer_10990swin_transformer_10992swin_transformer_10994swin_transformer_10996swin_transformer_10998swin_transformer_11000swin_transformer_11002swin_transformer_11004swin_transformer_11006swin_transformer_11008swin_transformer_11010*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *1
f,R*
(__inference_restored_function_body_10622?
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_11013swin_transformer_1_11015swin_transformer_1_11017swin_transformer_1_11019swin_transformer_1_11021swin_transformer_1_11023swin_transformer_1_11025swin_transformer_1_11027swin_transformer_1_11029swin_transformer_1_11031swin_transformer_1_11033swin_transformer_1_11035swin_transformer_1_11037swin_transformer_1_11039swin_transformer_1_11041*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *1
f,R*
(__inference_restored_function_body_10686?
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_11044*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9927?
(global_average_pooling1d/PartitionedCallPartitionedCall.patch_merging/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_10_11048dense_10_11050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_10098x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????W?
NoOpNoOp!^dense_10/StatefulPartitionedCall(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall$^random_crop/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2J
#random_crop/StatefulPartitionedCall#random_crop/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
E
)__inference_dropout_2_layer_call_fn_13199

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12230e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
C
'__inference_restored_function_body_9773

images
identity?
PartitionedCallPartitionedCallimages*
Tin
2*
Tout
2*,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_patch_extract_layer_call_and_return_conditional_losses_1291e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameimages
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_12230

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
6map_while_stateless_random_flip_left_right_false_12092u
qmap_while_stateless_random_flip_left_right_identity_map_while_stateless_random_flip_left_right_control_dependency7
3map_while_stateless_random_flip_left_right_identity?
3map/while/stateless_random_flip_left_right/IdentityIdentityqmap_while_stateless_random_flip_left_right_identity_map_while_stateless_random_flip_left_right_control_dependency*
T0*"
_output_shapes
:@@"s
3map_while_stateless_random_flip_left_right_identity<map/while/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:@@:( $
"
_output_shapes
:@@
?
?
 random_flip_map_while_cond_11595<
8random_flip_map_while_random_flip_map_while_loop_counter7
3random_flip_map_while_random_flip_map_strided_slice%
!random_flip_map_while_placeholder'
#random_flip_map_while_placeholder_1<
8random_flip_map_while_less_random_flip_map_strided_sliceS
Orandom_flip_map_while_random_flip_map_while_cond_11595___redundant_placeholder0S
Orandom_flip_map_while_random_flip_map_while_cond_11595___redundant_placeholder1"
random_flip_map_while_identity
?
random_flip/map/while/LessLess!random_flip_map_while_placeholder8random_flip_map_while_less_random_flip_map_strided_slice*
T0*
_output_shapes
: ?
random_flip/map/while/Less_1Less8random_flip_map_while_random_flip_map_while_loop_counter3random_flip_map_while_random_flip_map_strided_slice*
T0*
_output_shapes
: ?
 random_flip/map/while/LogicalAnd
LogicalAnd random_flip/map/while/Less_1:z:0random_flip/map/while/Less:z:0*
_output_shapes
: q
random_flip/map/while/IdentityIdentity$random_flip/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "I
random_flip_map_while_identity'random_flip/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
?	
a
E__inference_activation_layer_call_and_return_conditional_losses_12180

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*-
_output_shapes
:???????????P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???m
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*-
_output_shapes
:???????????Y
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:???????????O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:???????????e

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:???????????\
IdentityIdentityGelu/mul_1:z:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_dense_7_layer_call_and_return_conditional_losses_13260

inputs4
!tensordot_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????e
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:???????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_13661
file_prefixC
,assignvariableop_swin_transformer_1_variable:?5
"assignvariableop_1_dense_10_kernel:	?W.
 assignvariableop_2_dense_10_bias:WA
/assignvariableop_3_patch_embedding_dense_kernel:@;
-assignvariableop_4_patch_embedding_dense_bias:@J
7assignvariableop_5_patch_embedding_embedding_embeddings:	?@K
=assignvariableop_6_swin_transformer_layer_normalization_gamma:@J
<assignvariableop_7_swin_transformer_layer_normalization_beta:@M
;assignvariableop_8_swin_transformer_window_attention_weight:	V
Cassignvariableop_9_swin_transformer_window_attention_dense_1_kernel:	@?Q
Bassignvariableop_10_swin_transformer_window_attention_dense_1_bias:	?V
Dassignvariableop_11_swin_transformer_window_attention_dense_2_kernel:@@P
Bassignvariableop_12_swin_transformer_window_attention_dense_2_bias:@N
@assignvariableop_13_swin_transformer_layer_normalization_1_gamma:@M
?assignvariableop_14_swin_transformer_layer_normalization_1_beta:@5
"assignvariableop_15_dense_3_kernel:	@?/
 assignvariableop_16_dense_3_bias:	?5
"assignvariableop_17_dense_4_kernel:	?@.
 assignvariableop_18_dense_4_bias:@P
>assignvariableop_19_swin_transformer_window_attention_variable:	P
Bassignvariableop_20_swin_transformer_1_layer_normalization_2_gamma:@O
Aassignvariableop_21_swin_transformer_1_layer_normalization_2_beta:@R
@assignvariableop_22_swin_transformer_1_window_attention_1_weight:	[
Hassignvariableop_23_swin_transformer_1_window_attention_1_dense_5_kernel:	@?U
Fassignvariableop_24_swin_transformer_1_window_attention_1_dense_5_bias:	?Z
Hassignvariableop_25_swin_transformer_1_window_attention_1_dense_6_kernel:@@T
Fassignvariableop_26_swin_transformer_1_window_attention_1_dense_6_bias:@P
Bassignvariableop_27_swin_transformer_1_layer_normalization_3_gamma:@O
Aassignvariableop_28_swin_transformer_1_layer_normalization_3_beta:@5
"assignvariableop_29_dense_7_kernel:	@?/
 assignvariableop_30_dense_7_bias:	?5
"assignvariableop_31_dense_8_kernel:	?@.
 assignvariableop_32_dense_8_bias:@T
Bassignvariableop_33_swin_transformer_1_window_attention_1_variable:	D
0assignvariableop_34_patch_merging_dense_9_kernel:
??,
assignvariableop_35_statevar_1:	*
assignvariableop_36_statevar:	%
assignvariableop_37_total_2: %
assignvariableop_38_count_2: %
assignvariableop_39_total_1: %
assignvariableop_40_count_1: #
assignvariableop_41_total: #
assignvariableop_42_count: 
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B9layer_with_weights-2/attn_mask/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEBJlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp,assignvariableop_swin_transformer_1_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_10_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_10_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp/assignvariableop_3_patch_embedding_dense_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp-assignvariableop_4_patch_embedding_dense_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp7assignvariableop_5_patch_embedding_embedding_embeddingsIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp=assignvariableop_6_swin_transformer_layer_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp<assignvariableop_7_swin_transformer_layer_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp;assignvariableop_8_swin_transformer_window_attention_weightIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpCassignvariableop_9_swin_transformer_window_attention_dense_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpBassignvariableop_10_swin_transformer_window_attention_dense_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpDassignvariableop_11_swin_transformer_window_attention_dense_2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpBassignvariableop_12_swin_transformer_window_attention_dense_2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp@assignvariableop_13_swin_transformer_layer_normalization_1_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp?assignvariableop_14_swin_transformer_layer_normalization_1_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp assignvariableop_16_dense_3_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_4_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_4_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp>assignvariableop_19_swin_transformer_window_attention_variableIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpBassignvariableop_20_swin_transformer_1_layer_normalization_2_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpAassignvariableop_21_swin_transformer_1_layer_normalization_2_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp@assignvariableop_22_swin_transformer_1_window_attention_1_weightIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpHassignvariableop_23_swin_transformer_1_window_attention_1_dense_5_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpFassignvariableop_24_swin_transformer_1_window_attention_1_dense_5_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpHassignvariableop_25_swin_transformer_1_window_attention_1_dense_6_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpFassignvariableop_26_swin_transformer_1_window_attention_1_dense_6_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpBassignvariableop_27_swin_transformer_1_layer_normalization_3_gammaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpAassignvariableop_28_swin_transformer_1_layer_normalization_3_betaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp"assignvariableop_29_dense_7_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp assignvariableop_30_dense_7_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp"assignvariableop_31_dense_8_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp assignvariableop_32_dense_8_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpBassignvariableop_33_swin_transformer_1_window_attention_1_variableIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp0assignvariableop_34_patch_merging_dense_9_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpassignvariableop_35_statevar_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpassignvariableop_36_statevarIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_2Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
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
?
?
'__inference_dense_8_layer_call_fn_13313

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_12500t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?5
?
G__inference_patch_merging_layer_call_and_return_conditional_losses_2738
x=
)dense_9_tensordot_readvariableop_resource:
??
identity?? dense_9/Tensordot/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   g
ReshapeReshapexReshape/shape:output:0*
T0*/
_output_shapes
:?????????  @l
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ?
strided_sliceStridedSliceReshape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask	*
end_maskn
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"               p
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                p
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ?
strided_slice_1StridedSliceReshape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask	*
end_maskn
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"               p
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                p
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ?
strided_slice_2StridedSliceReshape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask	*
end_maskn
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"              p
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                p
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ?
strided_slice_3StridedSliceReshape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask	*
end_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2strided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0strided_slice_3:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????d
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      w
	Reshape_1Reshapeconcat:output:0Reshape_1/shape:output:0*
T0*-
_output_shapes
:????????????
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0`
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Y
dense_9/Tensordot/ShapeShapeReshape_1:output:0*
T0*
_output_shapes
:a
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_9/Tensordot/transpose	TransposeReshape_1:output:0!dense_9/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?a
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????i
NoOpNoOp!^dense_9/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentitydense_9/Tensordot:output:0^NoOp*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????@: 2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
?
/__inference_swin_transformer_layer_call_fn_2650
x
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	
	unknown_4:	
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:	@?

unknown_10:	?

unknown_11:	?@

unknown_12:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2631`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????@

_user_specified_namex
?r
?
cond_true_11854
cond_shape_inputs;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/Assert?$cond/stateful_uniform/RngReadAndSkipK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/sub/yConst*
_output_shapes
: *
dtype0*
value	B :@b
cond/subSubcond/strided_slice:output:0cond/sub/y:output:0*
T0*
_output_shapes
: m
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :@h

cond/sub_1Subcond/strided_slice_1:output:0cond/sub_1/y:output:0*
T0*
_output_shapes
: e
cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:[
cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : _
cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????e
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
cond/stateful_uniform/ProdProd$cond/stateful_uniform/shape:output:0$cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ^
cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :y
cond/stateful_uniform/Cast_1Cast#cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$cond/stateful_uniform/RngReadAndSkipRngReadAndSkip-cond_stateful_uniform_rngreadandskip_resource%cond/stateful_uniform/Cast/x:output:0 cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:s
)cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
cond/stateful_uniform/BitcastBitcast,cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0u
+cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
cond/stateful_uniformStatelessRandomUniformIntV2$cond/stateful_uniform/shape:output:0(cond/stateful_uniform/Bitcast_1:output:0&cond/stateful_uniform/Bitcast:output:0"cond/stateful_uniform/alg:output:0"cond/stateful_uniform/min:output:0"cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0d
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_2StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :U
cond/addAddV2cond/sub:z:0cond/add/y:output:0*
T0*
_output_shapes
: b
cond/modFloorModcond/strided_slice_2:output:0cond/add:z:0*
T0*
_output_shapes
: d
cond/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_3StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_3/stack:output:0%cond/strided_slice_3/stack_1:output:0%cond/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :[

cond/add_1AddV2cond/sub_1:z:0cond/add_1/y:output:0*
T0*
_output_shapes
: f

cond/mod_1FloorModcond/strided_slice_3:output:0cond/add_1:z:0*
T0*
_output_shapes
: `
cond/crop_to_bounding_box/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B :@?
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B :@?
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B :@?
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B :@?
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentitycond_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????@@c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B :@e
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B :@?
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????@@?
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:?????????@@?
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: 2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:5 1
/
_output_shapes
:?????????@@
?
?
*__inference_sequential_layer_call_fn_12372
dense_3_input
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_3_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12348t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:??????????@
'
_user_specified_namedense_3_input
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_12468

inputs

identity_1T
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????a

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?[
?
@__inference_model_layer_call_and_return_conditional_losses_11411

inputs'
patch_embedding_11331:@#
patch_embedding_11333:@(
patch_embedding_11335:	?@$
swin_transformer_11338:@$
swin_transformer_11340:@)
swin_transformer_11342:	@?%
swin_transformer_11344:	?(
swin_transformer_11346:	(
swin_transformer_11348:	(
swin_transformer_11350:@@$
swin_transformer_11352:@$
swin_transformer_11354:@$
swin_transformer_11356:@)
swin_transformer_11358:	@?%
swin_transformer_11360:	?)
swin_transformer_11362:	?@$
swin_transformer_11364:@&
swin_transformer_1_11367:@&
swin_transformer_1_11369:@+
swin_transformer_1_11371:	@?'
swin_transformer_1_11373:	?*
swin_transformer_1_11375:	*
swin_transformer_1_11377:	/
swin_transformer_1_11379:?*
swin_transformer_1_11381:@@&
swin_transformer_1_11383:@&
swin_transformer_1_11385:@&
swin_transformer_1_11387:@+
swin_transformer_1_11389:	@?'
swin_transformer_1_11391:	?+
swin_transformer_1_11393:	?@&
swin_transformer_1_11395:@'
patch_merging_11398:
??:
'dense_10_matmul_readvariableop_resource:	?W6
(dense_10_biasadd_readvariableop_resource:W
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?'patch_embedding/StatefulPartitionedCall?%patch_merging/StatefulPartitionedCall?(swin_transformer/StatefulPartitionedCall?*swin_transformer_1/StatefulPartitionedCallG
random_crop/ShapeShapeinputs*
T0*
_output_shapes
:r
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????t
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????k
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/strided_sliceStridedSlicerandom_crop/Shape:output:0(random_crop/strided_slice/stack:output:0*random_crop/strided_slice/stack_1:output:0*random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
!random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/strided_slice_1StridedSlicerandom_crop/Shape:output:0*random_crop/strided_slice_1/stack:output:0,random_crop/strided_slice_1/stack_1:output:0,random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskS
random_crop/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@y
random_crop/mulMul$random_crop/strided_slice_1:output:0random_crop/mul/y:output:0*
T0*
_output_shapes
: ]
random_crop/CastCastrandom_crop/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
random_crop/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Bu
random_crop/truedivRealDivrandom_crop/Cast:y:0random_crop/truediv/y:output:0*
T0*
_output_shapes
: c
random_crop/Cast_1Castrandom_crop/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: U
random_crop/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :@{
random_crop/mul_1Mul"random_crop/strided_slice:output:0random_crop/mul_1/y:output:0*
T0*
_output_shapes
: a
random_crop/Cast_2Castrandom_crop/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: \
random_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B{
random_crop/truediv_1RealDivrandom_crop/Cast_2:y:0 random_crop/truediv_1/y:output:0*
T0*
_output_shapes
: e
random_crop/Cast_3Castrandom_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: {
random_crop/MinimumMinimum"random_crop/strided_slice:output:0random_crop/Cast_1:y:0*
T0*
_output_shapes
: 
random_crop/Minimum_1Minimum$random_crop/strided_slice_1:output:0random_crop/Cast_3:y:0*
T0*
_output_shapes
: t
random_crop/subSub"random_crop/strided_slice:output:0random_crop/Minimum:z:0*
T0*
_output_shapes
: _
random_crop/Cast_4Castrandom_crop/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: \
random_crop/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @{
random_crop/truediv_2RealDivrandom_crop/Cast_4:y:0 random_crop/truediv_2/y:output:0*
T0*
_output_shapes
: e
random_crop/Cast_5Castrandom_crop/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: z
random_crop/sub_1Sub$random_crop/strided_slice_1:output:0random_crop/Minimum_1:z:0*
T0*
_output_shapes
: a
random_crop/Cast_6Castrandom_crop/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: \
random_crop/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @{
random_crop/truediv_3RealDivrandom_crop/Cast_6:y:0 random_crop/truediv_3/y:output:0*
T0*
_output_shapes
: e
random_crop/Cast_7Castrandom_crop/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: U
random_crop/stack/0Const*
_output_shapes
: *
dtype0*
value	B : U
random_crop/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/stackPackrandom_crop/stack/0:output:0random_crop/Cast_5:y:0random_crop/Cast_7:y:0random_crop/stack/3:output:0*
N*
T0*
_output_shapes
:`
random_crop/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????`
random_crop/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
random_crop/stack_1Packrandom_crop/stack_1/0:output:0random_crop/Minimum:z:0random_crop/Minimum_1:z:0random_crop/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
random_crop/SliceSliceinputsrandom_crop/stack:output:0random_crop/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????@@?????????h
random_crop/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   ?
!random_crop/resize/ResizeBilinearResizeBilinearrandom_crop/Slice:output:0 random_crop/resize/size:output:0*
T0*/
_output_shapes
:?????????@@*
half_pixel_centers(?
patch_extract/PartitionedCallPartitionedCall2random_crop/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9773?
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_11331patch_embedding_11333patch_embedding_11335*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9785?
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_11338swin_transformer_11340swin_transformer_11342swin_transformer_11344swin_transformer_11346swin_transformer_11348swin_transformer_11350swin_transformer_11352swin_transformer_11354swin_transformer_11356swin_transformer_11358swin_transformer_11360swin_transformer_11362swin_transformer_11364*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9825?
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_11367swin_transformer_1_11369swin_transformer_1_11371swin_transformer_1_11373swin_transformer_1_11375swin_transformer_1_11377swin_transformer_1_11379swin_transformer_1_11381swin_transformer_1_11383swin_transformer_1_11385swin_transformer_1_11387swin_transformer_1_11389swin_transformer_1_11391swin_transformer_1_11393swin_transformer_1_11395*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9889?
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_11398*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9927q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMean.patch_merging/StatefulPartitionedCall:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	?W*
dtype0?
dense_10/MatMulMatMul&global_average_pooling1d/Mean:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????W?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:W*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Wh
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Wi
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????W?
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_2354
xI
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@O
<window_attention_1_dense_5_tensordot_readvariableop_resource:	@?I
:window_attention_1_dense_5_biasadd_readvariableop_resource:	?F
4window_attention_1_reshape_1_readvariableop_resource:	4
"window_attention_1_gather_resource:	N
7window_attention_1_expanddims_1_readvariableop_resource:?N
<window_attention_1_dense_6_tensordot_readvariableop_resource:@@H
:window_attention_1_dense_6_biasadd_readvariableop_resource:@I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_3_batchnorm_readvariableop_resource:@I
6sequential_1_dense_7_tensordot_readvariableop_resource:	@?C
4sequential_1_dense_7_biasadd_readvariableop_resource:	?I
6sequential_1_dense_8_tensordot_readvariableop_resource:	?@B
4sequential_1_dense_8_biasadd_readvariableop_resource:@
identity??.layer_normalization_2/batchnorm/ReadVariableOp?2layer_normalization_2/batchnorm/mul/ReadVariableOp?.layer_normalization_3/batchnorm/ReadVariableOp?2layer_normalization_3/batchnorm/mul/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?-sequential_1/dense_7/Tensordot/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?-sequential_1/dense_8/Tensordot/ReadVariableOp?.window_attention_1/ExpandDims_1/ReadVariableOp?window_attention_1/Gather?+window_attention_1/Reshape_1/ReadVariableOp?1window_attention_1/dense_5/BiasAdd/ReadVariableOp?3window_attention_1/dense_5/Tensordot/ReadVariableOp?1window_attention_1/dense_6/BiasAdd/ReadVariableOp?3window_attention_1/dense_6/Tensordot/ReadVariableOp~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_2/moments/meanMeanx=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencex3layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/mul_1Mulx'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   ?
ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:?????????  @[

Roll/shiftConst*
_output_shapes
:*
dtype0*
valueB"????????Z
	Roll/axisConst*
_output_shapes
:*
dtype0*
valueB"      ?
RollRollReshape:output:0Roll/shift:output:0Roll/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:?????????  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   
	Reshape_1ReshapeRoll:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!?????????@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
3window_attention_1/dense_5/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0s
)window_attention_1/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)window_attention_1/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
*window_attention_1/dense_5/Tensordot/ShapeShapeReshape_3:output:0*
T0*
_output_shapes
:t
2window_attention_1/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_5/Tensordot/GatherV2GatherV23window_attention_1/dense_5/Tensordot/Shape:output:02window_attention_1/dense_5/Tensordot/free:output:0;window_attention_1/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4window_attention_1/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/window_attention_1/dense_5/Tensordot/GatherV2_1GatherV23window_attention_1/dense_5/Tensordot/Shape:output:02window_attention_1/dense_5/Tensordot/axes:output:0=window_attention_1/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*window_attention_1/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention_1/dense_5/Tensordot/ProdProd6window_attention_1/dense_5/Tensordot/GatherV2:output:03window_attention_1/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
+window_attention_1/dense_5/Tensordot/Prod_1Prod8window_attention_1/dense_5/Tensordot/GatherV2_1:output:05window_attention_1/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention_1/dense_5/Tensordot/concatConcatV22window_attention_1/dense_5/Tensordot/free:output:02window_attention_1/dense_5/Tensordot/axes:output:09window_attention_1/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*window_attention_1/dense_5/Tensordot/stackPack2window_attention_1/dense_5/Tensordot/Prod:output:04window_attention_1/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
.window_attention_1/dense_5/Tensordot/transpose	TransposeReshape_3:output:04window_attention_1/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
,window_attention_1/dense_5/Tensordot/ReshapeReshape2window_attention_1/dense_5/Tensordot/transpose:y:03window_attention_1/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
+window_attention_1/dense_5/Tensordot/MatMulMatMul5window_attention_1/dense_5/Tensordot/Reshape:output:0;window_attention_1/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
,window_attention_1/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?t
2window_attention_1/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_5/Tensordot/concat_1ConcatV26window_attention_1/dense_5/Tensordot/GatherV2:output:05window_attention_1/dense_5/Tensordot/Const_2:output:0;window_attention_1/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
$window_attention_1/dense_5/TensordotReshape5window_attention_1/dense_5/Tensordot/MatMul:product:06window_attention_1/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:???????????
1window_attention_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"window_attention_1/dense_5/BiasAddBiasAdd-window_attention_1/dense_5/Tensordot:output:09window_attention_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????}
 window_attention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention_1/ReshapeReshape+window_attention_1/dense_5/BiasAdd:output:0)window_attention_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:?????????~
!window_attention_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
window_attention_1/transpose	Transpose#window_attention_1/Reshape:output:0*window_attention_1/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????p
&window_attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(window_attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention_1/strided_sliceStridedSlice window_attention_1/transpose:y:0/window_attention_1/strided_slice/stack:output:01window_attention_1/strided_slice/stack_1:output:01window_attention_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskr
(window_attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"window_attention_1/strided_slice_1StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_1/stack:output:03window_attention_1/strided_slice_1/stack_1:output:03window_attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskr
(window_attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"window_attention_1/strided_slice_2StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_2/stack:output:03window_attention_1/strided_slice_2/stack_1:output:03window_attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_mask]
window_attention_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
window_attention_1/mulMul)window_attention_1/strided_slice:output:0!window_attention_1/mul/y:output:0*
T0*/
_output_shapes
:?????????|
#window_attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention_1/transpose_1	Transpose+window_attention_1/strided_slice_1:output:0,window_attention_1/transpose_1/perm:output:0*
T0*/
_output_shapes
:??????????
window_attention_1/matmulBatchMatMulV2window_attention_1/mul:z:0"window_attention_1/transpose_1:y:0*
T0*/
_output_shapes
:??????????
+window_attention_1/Reshape_1/ReadVariableOpReadVariableOp4window_attention_1_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	u
"window_attention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
window_attention_1/Reshape_1Reshape3window_attention_1/Reshape_1/ReadVariableOp:value:0+window_attention_1/Reshape_1/shape:output:0*
T0	*
_output_shapes
:?
window_attention_1/GatherResourceGather"window_attention_1_gather_resource%window_attention_1/Reshape_1:output:0*
Tindices0	*
_output_shapes

:*
dtype0t
window_attention_1/IdentityIdentity"window_attention_1/Gather:output:0*
T0*
_output_shapes

:w
"window_attention_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?????
window_attention_1/Reshape_2Reshape$window_attention_1/Identity:output:0+window_attention_1/Reshape_2/shape:output:0*
T0*"
_output_shapes
:x
#window_attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
window_attention_1/transpose_2	Transpose%window_attention_1/Reshape_2:output:0,window_attention_1/transpose_2/perm:output:0*
T0*"
_output_shapes
:c
!window_attention_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention_1/ExpandDims
ExpandDims"window_attention_1/transpose_2:y:0*window_attention_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:?
window_attention_1/addAddV2"window_attention_1/matmul:output:0&window_attention_1/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
.window_attention_1/ExpandDims_1/ReadVariableOpReadVariableOp7window_attention_1_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype0e
#window_attention_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
window_attention_1/ExpandDims_1
ExpandDims6window_attention_1/ExpandDims_1/ReadVariableOp:value:0,window_attention_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?e
#window_attention_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention_1/ExpandDims_2
ExpandDims(window_attention_1/ExpandDims_1:output:0,window_attention_1/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:??
window_attention_1/CastCast(window_attention_1/ExpandDims_2:output:0*

DstT0*

SrcT0*+
_output_shapes
:?
"window_attention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention_1/Reshape_3Reshapewindow_attention_1/add:z:0+window_attention_1/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????
window_attention_1/add_1AddV2%window_attention_1/Reshape_3:output:0window_attention_1/Cast:y:0*
T0*4
_output_shapes"
 :??????????{
"window_attention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
window_attention_1/Reshape_4Reshapewindow_attention_1/add_1:z:0+window_attention_1/Reshape_4/shape:output:0*
T0*/
_output_shapes
:??????????
window_attention_1/SoftmaxSoftmax%window_attention_1/Reshape_4:output:0*
T0*/
_output_shapes
:??????????
%window_attention_1/dropout_3/IdentityIdentity$window_attention_1/Softmax:softmax:0*
T0*/
_output_shapes
:??????????
window_attention_1/matmul_1BatchMatMulV2.window_attention_1/dropout_3/Identity:output:0+window_attention_1/strided_slice_2:output:0*
T0*/
_output_shapes
:?????????|
#window_attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention_1/transpose_3	Transpose$window_attention_1/matmul_1:output:0,window_attention_1/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????w
"window_attention_1/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   ?
window_attention_1/Reshape_5Reshape"window_attention_1/transpose_3:y:0+window_attention_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????@?
3window_attention_1/dense_6/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_6_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0s
)window_attention_1/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)window_attention_1/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
*window_attention_1/dense_6/Tensordot/ShapeShape%window_attention_1/Reshape_5:output:0*
T0*
_output_shapes
:t
2window_attention_1/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_6/Tensordot/GatherV2GatherV23window_attention_1/dense_6/Tensordot/Shape:output:02window_attention_1/dense_6/Tensordot/free:output:0;window_attention_1/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4window_attention_1/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/window_attention_1/dense_6/Tensordot/GatherV2_1GatherV23window_attention_1/dense_6/Tensordot/Shape:output:02window_attention_1/dense_6/Tensordot/axes:output:0=window_attention_1/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*window_attention_1/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention_1/dense_6/Tensordot/ProdProd6window_attention_1/dense_6/Tensordot/GatherV2:output:03window_attention_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
+window_attention_1/dense_6/Tensordot/Prod_1Prod8window_attention_1/dense_6/Tensordot/GatherV2_1:output:05window_attention_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention_1/dense_6/Tensordot/concatConcatV22window_attention_1/dense_6/Tensordot/free:output:02window_attention_1/dense_6/Tensordot/axes:output:09window_attention_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*window_attention_1/dense_6/Tensordot/stackPack2window_attention_1/dense_6/Tensordot/Prod:output:04window_attention_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
.window_attention_1/dense_6/Tensordot/transpose	Transpose%window_attention_1/Reshape_5:output:04window_attention_1/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
,window_attention_1/dense_6/Tensordot/ReshapeReshape2window_attention_1/dense_6/Tensordot/transpose:y:03window_attention_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
+window_attention_1/dense_6/Tensordot/MatMulMatMul5window_attention_1/dense_6/Tensordot/Reshape:output:0;window_attention_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
,window_attention_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2window_attention_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_6/Tensordot/concat_1ConcatV26window_attention_1/dense_6/Tensordot/GatherV2:output:05window_attention_1/dense_6/Tensordot/Const_2:output:0;window_attention_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
$window_attention_1/dense_6/TensordotReshape5window_attention_1/dense_6/Tensordot/MatMul:product:06window_attention_1/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"window_attention_1/dense_6/BiasAddBiasAdd-window_attention_1/dense_6/Tensordot:output:09window_attention_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@?
'window_attention_1/dropout_3/Identity_1Identity+window_attention_1/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   ?
	Reshape_4Reshape0window_attention_1/dropout_3/Identity_1:output:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:?????????@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!?????????@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:?????????  @]
Roll_1/shiftConst*
_output_shapes
:*
dtype0*
valueB"      \
Roll_1/axisConst*
_output_shapes
:*
dtype0*
valueB"      ?
Roll_1RollReshape_6:output:0Roll_1/shift:output:0Roll_1/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:?????????  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   v
	Reshape_7ReshapeRoll_1:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:??????????@S
drop_path_1/ShapeShapeReshape_7:output:0*
T0*
_output_shapes
:i
drop_path_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!drop_path_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!drop_path_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path_1/strided_sliceStridedSlicedrop_path_1/Shape:output:0(drop_path_1/strided_slice/stack:output:0*drop_path_1/strided_slice/stack_1:output:0*drop_path_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"drop_path_1/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d
"drop_path_1/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
 drop_path_1/random_uniform/shapePack"drop_path_1/strided_slice:output:0+drop_path_1/random_uniform/shape/1:output:0+drop_path_1/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:?
(drop_path_1/random_uniform/RandomUniformRandomUniform)drop_path_1/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0V
drop_path_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/addAddV2drop_path_1/add/x:output:01drop_path_1/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:?????????e
drop_path_1/FloorFloordrop_path_1/add:z:0*
T0*+
_output_shapes
:?????????Z
drop_path_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/truedivRealDivReshape_7:output:0drop_path_1/truediv/y:output:0*
T0*,
_output_shapes
:??????????@}
drop_path_1/mulMuldrop_path_1/truediv:z:0drop_path_1/Floor:y:0*
T0*,
_output_shapes
:??????????@[
addAddV2xdrop_path_1/mul:z:0*
T0*,
_output_shapes
:??????????@~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_3/moments/meanMeanadd:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/mul_1Muladd:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@?
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0m
#sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential_1/dense_7/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_7/Tensordot/GatherV2GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/free:output:05sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/dense_7/Tensordot/GatherV2_1GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/axes:output:07sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????i
$sequential_1/activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"sequential_1/activation_1/Gelu/mulMul-sequential_1/activation_1/Gelu/mul/x:output:0%sequential_1/dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:???????????j
%sequential_1/activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
&sequential_1/activation_1/Gelu/truedivRealDiv%sequential_1/dense_7/BiasAdd:output:0.sequential_1/activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:????????????
"sequential_1/activation_1/Gelu/ErfErf*sequential_1/activation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????i
$sequential_1/activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"sequential_1/activation_1/Gelu/addAddV2-sequential_1/activation_1/Gelu/add/x:output:0&sequential_1/activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
$sequential_1/activation_1/Gelu/mul_1Mul&sequential_1/activation_1/Gelu/mul:z:0&sequential_1/activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:????????????
sequential_1/dropout_4/IdentityIdentity(sequential_1/activation_1/Gelu/mul_1:z:0*
T0*-
_output_shapes
:????????????
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0m
#sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
$sequential_1/dense_8/Tensordot/ShapeShape(sequential_1/dropout_4/Identity:output:0*
T0*
_output_shapes
:n
,sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_8/Tensordot/GatherV2GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/free:output:05sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/dense_8/Tensordot/GatherV2_1GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/axes:output:07sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
(sequential_1/dense_8/Tensordot/transpose	Transpose(sequential_1/dropout_4/Identity:output:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
sequential_1/dropout_5/IdentityIdentity%sequential_1/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@k
drop_path_1/Shape_1Shape(sequential_1/dropout_5/Identity:output:0*
T0*
_output_shapes
:k
!drop_path_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#drop_path_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#drop_path_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path_1/strided_slice_1StridedSlicedrop_path_1/Shape_1:output:0*drop_path_1/strided_slice_1/stack:output:0,drop_path_1/strided_slice_1/stack_1:output:0,drop_path_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$drop_path_1/random_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :f
$drop_path_1/random_uniform_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
"drop_path_1/random_uniform_1/shapePack$drop_path_1/strided_slice_1:output:0-drop_path_1/random_uniform_1/shape/1:output:0-drop_path_1/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
*drop_path_1/random_uniform_1/RandomUniformRandomUniform+drop_path_1/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0X
drop_path_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/add_1AddV2drop_path_1/add_1/x:output:03drop_path_1/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:?????????i
drop_path_1/Floor_1Floordrop_path_1/add_1:z:0*
T0*+
_output_shapes
:?????????\
drop_path_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/truediv_1RealDiv(sequential_1/dropout_5/Identity:output:0 drop_path_1/truediv_1/y:output:0*
T0*,
_output_shapes
:??????????@?
drop_path_1/mul_1Muldrop_path_1/truediv_1:z:0drop_path_1/Floor_1:y:0*
T0*,
_output_shapes
:??????????@e
add_1AddV2add:z:0drop_path_1/mul_1:z:0*
T0*,
_output_shapes
:??????????@?
NoOpNoOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp/^window_attention_1/ExpandDims_1/ReadVariableOp^window_attention_1/Gather,^window_attention_1/Reshape_1/ReadVariableOp2^window_attention_1/dense_5/BiasAdd/ReadVariableOp4^window_attention_1/dense_5/Tensordot/ReadVariableOp2^window_attention_1/dense_6/BiasAdd/ReadVariableOp4^window_attention_1/dense_6/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????@: : : : : : : : : : : : : : : 2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2^
-sequential_1/dense_7/Tensordot/ReadVariableOp-sequential_1/dense_7/Tensordot/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2^
-sequential_1/dense_8/Tensordot/ReadVariableOp-sequential_1/dense_8/Tensordot/ReadVariableOp2`
.window_attention_1/ExpandDims_1/ReadVariableOp.window_attention_1/ExpandDims_1/ReadVariableOp26
window_attention_1/Gatherwindow_attention_1/Gather2Z
+window_attention_1/Reshape_1/ReadVariableOp+window_attention_1/Reshape_1/ReadVariableOp2f
1window_attention_1/dense_5/BiasAdd/ReadVariableOp1window_attention_1/dense_5/BiasAdd/ReadVariableOp2j
3window_attention_1/dense_5/Tensordot/ReadVariableOp3window_attention_1/dense_5/Tensordot/ReadVariableOp2f
1window_attention_1/dense_6/BiasAdd/ReadVariableOp1window_attention_1/dense_6/BiasAdd/ReadVariableOp2j
3window_attention_1/dense_6/Tensordot/ReadVariableOp3window_attention_1/dense_6/Tensordot/ReadVariableOp:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
?
6map_while_stateless_random_flip_left_right_false_10272u
qmap_while_stateless_random_flip_left_right_identity_map_while_stateless_random_flip_left_right_control_dependency7
3map_while_stateless_random_flip_left_right_identity?
3map/while/stateless_random_flip_left_right/IdentityIdentityqmap_while_stateless_random_flip_left_right_identity_map_while_stateless_random_flip_left_right_control_dependency*
T0*"
_output_shapes
:@@"s
3map_while_stateless_random_flip_left_right_identity<map/while/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:@@:( $
"
_output_shapes
:@@
?V
?
E__inference_sequential_layer_call_and_return_conditional_losses_12900

inputs<
)dense_3_tensordot_readvariableop_resource:	@?6
'dense_3_biasadd_readvariableop_resource:	?<
)dense_4_tensordot_readvariableop_resource:	?@5
'dense_4_biasadd_readvariableop_resource:@
identity??dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       M
dense_3/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_3/Tensordot/transpose	Transposeinputs!dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????Z
activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
activation/Gelu/mulMulactivation/Gelu/mul/x:output:0dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:???????????[
activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
activation/Gelu/truedivRealDivdense_3/BiasAdd:output:0activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:???????????o
activation/Gelu/ErfErfactivation/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????Z
activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
activation/Gelu/addAddV2activation/Gelu/add/x:output:0activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
activation/Gelu/mul_1Mulactivation/Gelu/mul:z:0activation/Gelu/add:z:0*
T0*-
_output_shapes
:???????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
dropout_1/dropout/MulMulactivation/Gelu/mul_1:z:0 dropout_1/dropout/Const:output:0*
T0*-
_output_shapes
:???????????`
dropout_1/dropout/ShapeShapeactivation/Gelu/mul_1:z:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*-
_output_shapes
:????????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_4/Tensordot/ShapeShapedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/transpose	Transposedropout_1/dropout/Mul_1:z:0!dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
dropout_2/dropout/MulMuldense_4/BiasAdd:output:0 dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:??????????@_
dropout_2/dropout/ShapeShapedense_4/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@o
IdentityIdentitydropout_2/dropout/Mul_1:z:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
c
G__inference_patch_extract_layer_call_and_return_conditional_losses_3697

images
identity;
ShapeShapeimages*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
ExtractImagePatchesExtractImagePatchesimages*
T0*/
_output_shapes
:?????????  *
ksizes
*
paddingVALID*
rates
*
strides
R
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*,
_output_shapes
:??????????]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameimages
?

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_13221

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q???i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?e
?
 random_flip_map_while_body_11596<
8random_flip_map_while_random_flip_map_while_loop_counter7
3random_flip_map_while_random_flip_map_strided_slice%
!random_flip_map_while_placeholder'
#random_flip_map_while_placeholder_1;
7random_flip_map_while_random_flip_map_strided_slice_1_0w
srandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensor_0W
Irandom_flip_map_while_stateful_uniform_full_int_rngreadandskip_resource_0:	"
random_flip_map_while_identity$
 random_flip_map_while_identity_1$
 random_flip_map_while_identity_2$
 random_flip_map_while_identity_39
5random_flip_map_while_random_flip_map_strided_slice_1u
qrandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensorU
Grandom_flip_map_while_stateful_uniform_full_int_rngreadandskip_resource:	??>random_flip/map/while/stateful_uniform_full_int/RngReadAndSkip?
Grandom_flip/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      ?
9random_flip/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsrandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensor_0!random_flip_map_while_placeholderPrandom_flip/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*"
_output_shapes
:@@*
element_dtype0
5random_flip/map/while/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:
5random_flip/map/while/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4random_flip/map/while/stateful_uniform_full_int/ProdProd>random_flip/map/while/stateful_uniform_full_int/shape:output:0>random_flip/map/while/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: x
6random_flip/map/while/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
6random_flip/map/while/stateful_uniform_full_int/Cast_1Cast=random_flip/map/while/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
>random_flip/map/while/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipIrandom_flip_map_while_stateful_uniform_full_int_rngreadandskip_resource_0?random_flip/map/while/stateful_uniform_full_int/Cast/x:output:0:random_flip/map/while/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:?
Crandom_flip/map/while/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Erandom_flip/map/while/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Erandom_flip/map/while/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=random_flip/map/while/stateful_uniform_full_int/strided_sliceStridedSliceFrandom_flip/map/while/stateful_uniform_full_int/RngReadAndSkip:value:0Lrandom_flip/map/while/stateful_uniform_full_int/strided_slice/stack:output:0Nrandom_flip/map/while/stateful_uniform_full_int/strided_slice/stack_1:output:0Nrandom_flip/map/while/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
7random_flip/map/while/stateful_uniform_full_int/BitcastBitcastFrandom_flip/map/while/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
Erandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Grandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Grandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?random_flip/map/while/stateful_uniform_full_int/strided_slice_1StridedSliceFrandom_flip/map/while/stateful_uniform_full_int/RngReadAndSkip:value:0Nrandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stack:output:0Prandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Prandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
9random_flip/map/while/stateful_uniform_full_int/Bitcast_1BitcastHrandom_flip/map/while/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0u
3random_flip/map/while/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
/random_flip/map/while/stateful_uniform_full_intStatelessRandomUniformFullIntV2>random_flip/map/while/stateful_uniform_full_int/shape:output:0Brandom_flip/map/while/stateful_uniform_full_int/Bitcast_1:output:0@random_flip/map/while/stateful_uniform_full_int/Bitcast:output:0<random_flip/map/while/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	j
 random_flip/map/while/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ?
random_flip/map/while/stackPack8random_flip/map/while/stateful_uniform_full_int:output:0)random_flip/map/while/zeros_like:output:0*
N*
T0	*
_output_shapes

:z
)random_flip/map/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        |
+random_flip/map/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       |
+random_flip/map/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
#random_flip/map/while/strided_sliceStridedSlice$random_flip/map/while/stack:output:02random_flip/map/while/strided_slice/stack:output:04random_flip/map/while/strided_slice/stack_1:output:04random_flip/map/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask?
Irandom_flip/map/while/stateless_random_flip_left_right/control_dependencyIdentity@random_flip/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*L
_classB
@>loc:@random_flip/map/while/TensorArrayV2Read/TensorListGetItem*"
_output_shapes
:@@?
Urandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Srandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Srandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter,random_flip/map/while/strided_slice:output:0* 
_output_shapes
::?
lrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
hrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2^random_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0rrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0vrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0urandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ?
Srandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/subSub\random_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/max:output:0\random_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
Srandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/mulMulqrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Wrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ?
Orandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniformAddV2Wrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0\random_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
=random_flip/map/while/stateless_random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
;random_flip/map/while/stateless_random_flip_left_right/LessLessSrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform:z:0Frandom_flip/map/while/stateless_random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: ?
6random_flip/map/while/stateless_random_flip_left_rightStatelessIf?random_flip/map/while/stateless_random_flip_left_right/Less:z:0Rrandom_flip/map/while/stateless_random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*"
_output_shapes
:@@* 
_read_only_resource_inputs
 *U
else_branchFRD
Brandom_flip_map_while_stateless_random_flip_left_right_false_11656*!
output_shapes
:@@*T
then_branchERC
Arandom_flip_map_while_stateless_random_flip_left_right_true_11655?
?random_flip/map/while/stateless_random_flip_left_right/IdentityIdentity?random_flip/map/while/stateless_random_flip_left_right:output:0*
T0*"
_output_shapes
:@@?
:random_flip/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#random_flip_map_while_placeholder_1!random_flip_map_while_placeholderHrandom_flip/map/while/stateless_random_flip_left_right/Identity:output:0*
_output_shapes
: *
element_dtype0:???]
random_flip/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
random_flip/map/while/addAddV2!random_flip_map_while_placeholder$random_flip/map/while/add/y:output:0*
T0*
_output_shapes
: _
random_flip/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
random_flip/map/while/add_1AddV28random_flip_map_while_random_flip_map_while_loop_counter&random_flip/map/while/add_1/y:output:0*
T0*
_output_shapes
: ?
random_flip/map/while/IdentityIdentityrandom_flip/map/while/add_1:z:0^random_flip/map/while/NoOp*
T0*
_output_shapes
: ?
 random_flip/map/while/Identity_1Identity3random_flip_map_while_random_flip_map_strided_slice^random_flip/map/while/NoOp*
T0*
_output_shapes
: ?
 random_flip/map/while/Identity_2Identityrandom_flip/map/while/add:z:0^random_flip/map/while/NoOp*
T0*
_output_shapes
: ?
 random_flip/map/while/Identity_3IdentityJrandom_flip/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^random_flip/map/while/NoOp*
T0*
_output_shapes
: ?
random_flip/map/while/NoOpNoOp?^random_flip/map/while/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "I
random_flip_map_while_identity'random_flip/map/while/Identity:output:0"M
 random_flip_map_while_identity_1)random_flip/map/while/Identity_1:output:0"M
 random_flip_map_while_identity_2)random_flip/map/while/Identity_2:output:0"M
 random_flip_map_while_identity_3)random_flip/map/while/Identity_3:output:0"p
5random_flip_map_while_random_flip_map_strided_slice_17random_flip_map_while_random_flip_map_strided_slice_1_0"?
Grandom_flip_map_while_stateful_uniform_full_int_rngreadandskip_resourceIrandom_flip_map_while_stateful_uniform_full_int_rngreadandskip_resource_0"?
qrandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensorsrandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2?
>random_flip/map/while/stateful_uniform_full_int/RngReadAndSkip>random_flip/map/while/stateful_uniform_full_int/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
G
+__inference_random_flip_layer_call_fn_12005

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10013h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_12348

inputs 
dense_3_12334:	@?
dense_3_12336:	? 
dense_4_12341:	?@
dense_4_12343:@
identity??dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_12334dense_3_12336*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_12162?
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12180?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12297?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_4_12341dense_4_12343*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_12219?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12264~
IdentityIdentity*dropout_2/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
H
,__inference_activation_1_layer_call_fn_13265

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_12461f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
c
G__inference_activation_1_layer_call_and_return_conditional_losses_12461

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*-
_output_shapes
:???????????P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???m
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*-
_output_shapes
:???????????Y
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:???????????O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:???????????e

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:???????????\
IdentityIdentityGelu/mul_1:z:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_swin_transformer_layer_call_fn_6442
x
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	
	unknown_4:	
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:	@?

unknown_10:	?

unknown_11:	?@

unknown_12:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_swin_transformer_layer_call_and_return_conditional_losses_6423`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????@

_user_specified_namex
?(
?
I__inference_patch_embedding_layer_call_and_return_conditional_losses_1264	
patch9
'dense_tensordot_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@4
!embedding_embedding_lookup_368004:	?@
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?embedding/embedding_lookupM
range/startConst*
_output_shapes
: *
dtype0*
value	B : N
range/limitConst*
_output_shapes
: *
dtype0*
value
B :?M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :m
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:??
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       J
dense/Tensordot/ShapeShapepatch*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transposepatchdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:???????????
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_368004range:output:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/368004*
_output_shapes
:	?@*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/368004*
_output_shapes
:	?@?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	?@?
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????@?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 [
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:S O
,
_output_shapes
:??????????

_user_specified_namepatch
?
?
,__inference_sequential_1_layer_call_fn_12525
dense_7_input
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_12514t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:??????????@
'
_user_specified_namedense_7_input
?&
R
cond_false_11855
cond_shape_inputs
cond_placeholder
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :@f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Bf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????@@?????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:?????????@@*
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????@@"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: :5 1
/
_output_shapes
:?????????@@
?V
?
map_while_body_10212$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0K
=map_while_stateful_uniform_full_int_rngreadandskip_resource_0:	
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorI
;map_while_stateful_uniform_full_int_rngreadandskip_resource:	??2map/while/stateful_uniform_full_int/RngReadAndSkip?
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      ?
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*"
_output_shapes
:@@*
element_dtype0s
)map/while/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:s
)map/while/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
(map/while/stateful_uniform_full_int/ProdProd2map/while/stateful_uniform_full_int/shape:output:02map/while/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: l
*map/while/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
*map/while/stateful_uniform_full_int/Cast_1Cast1map/while/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
2map/while/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=map_while_stateful_uniform_full_int_rngreadandskip_resource_03map/while/stateful_uniform_full_int/Cast/x:output:0.map/while/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:?
7map/while/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9map/while/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9map/while/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1map/while/stateful_uniform_full_int/strided_sliceStridedSlice:map/while/stateful_uniform_full_int/RngReadAndSkip:value:0@map/while/stateful_uniform_full_int/strided_slice/stack:output:0Bmap/while/stateful_uniform_full_int/strided_slice/stack_1:output:0Bmap/while/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
+map/while/stateful_uniform_full_int/BitcastBitcast:map/while/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
9map/while/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
;map/while/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;map/while/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3map/while/stateful_uniform_full_int/strided_slice_1StridedSlice:map/while/stateful_uniform_full_int/RngReadAndSkip:value:0Bmap/while/stateful_uniform_full_int/strided_slice_1/stack:output:0Dmap/while/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Dmap/while/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
-map/while/stateful_uniform_full_int/Bitcast_1Bitcast<map/while/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'map/while/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
#map/while/stateful_uniform_full_intStatelessRandomUniformFullIntV22map/while/stateful_uniform_full_int/shape:output:06map/while/stateful_uniform_full_int/Bitcast_1:output:04map/while/stateful_uniform_full_int/Bitcast:output:00map/while/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	^
map/while/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ?
map/while/stackPack,map/while/stateful_uniform_full_int:output:0map/while/zeros_like:output:0*
N*
T0	*
_output_shapes

:n
map/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
map/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
map/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
map/while/strided_sliceStridedSlicemap/while/stack:output:0&map/while/strided_slice/stack:output:0(map/while/strided_slice/stack_1:output:0(map/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask?
=map/while/stateless_random_flip_left_right/control_dependencyIdentity4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*@
_class6
42loc:@map/while/TensorArrayV2Read/TensorListGetItem*"
_output_shapes
:@@?
Imap/while/stateless_random_flip_left_right/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
`map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter map/while/strided_slice:output:0* 
_output_shapes
::?
`map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
\map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Rmap/while/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0fmap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0jmap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0imap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ?
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/subSubPmap/while/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Pmap/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/mulMulemap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Kmap/while/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ?
Cmap/while/stateless_random_flip_left_right/stateless_random_uniformAddV2Kmap/while/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Pmap/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: v
1map/while/stateless_random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
/map/while/stateless_random_flip_left_right/LessLessGmap/while/stateless_random_flip_left_right/stateless_random_uniform:z:0:map/while/stateless_random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: ?
*map/while/stateless_random_flip_left_rightStatelessIf3map/while/stateless_random_flip_left_right/Less:z:0Fmap/while/stateless_random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*"
_output_shapes
:@@* 
_read_only_resource_inputs
 *I
else_branch:R8
6map_while_stateless_random_flip_left_right_false_10272*!
output_shapes
:@@*H
then_branch9R7
5map_while_stateless_random_flip_left_right_true_10271?
3map/while/stateless_random_flip_left_right/IdentityIdentity3map/while/stateless_random_flip_left_right:output:0*
T0*"
_output_shapes
:@@?
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder<map/while/stateless_random_flip_left_right/Identity:output:0*
_output_shapes
: *
element_dtype0:???Q
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: e
map/while/IdentityIdentitymap/while/add_1:z:0^map/while/NoOp*
T0*
_output_shapes
: o
map/while/Identity_1Identitymap_while_map_strided_slice^map/while/NoOp*
T0*
_output_shapes
: e
map/while/Identity_2Identitymap/while/add:z:0^map/while/NoOp*
T0*
_output_shapes
: ?
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: ?
map/while/NoOpNoOp3^map/while/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"|
;map_while_stateful_uniform_full_int_rngreadandskip_resource=map_while_stateful_uniform_full_int_rngreadandskip_resource_0"?
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2h
2map/while/stateful_uniform_full_int/RngReadAndSkip2map/while/stateful_uniform_full_int/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_12187

inputs

identity_1T
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????a

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?9
?
@__inference_model_layer_call_and_return_conditional_losses_10728

inputs
random_crop_10576:	
random_flip_10579:	'
patch_embedding_10583:@#
patch_embedding_10585:@(
patch_embedding_10587:	?@$
swin_transformer_10623:@$
swin_transformer_10625:@)
swin_transformer_10627:	@?%
swin_transformer_10629:	?(
swin_transformer_10631:	(
swin_transformer_10633:	(
swin_transformer_10635:@@$
swin_transformer_10637:@$
swin_transformer_10639:@$
swin_transformer_10641:@)
swin_transformer_10643:	@?%
swin_transformer_10645:	?)
swin_transformer_10647:	?@$
swin_transformer_10649:@&
swin_transformer_1_10687:@&
swin_transformer_1_10689:@+
swin_transformer_1_10691:	@?'
swin_transformer_1_10693:	?*
swin_transformer_1_10695:	*
swin_transformer_1_10697:	/
swin_transformer_1_10699:?*
swin_transformer_1_10701:@@&
swin_transformer_1_10703:@&
swin_transformer_1_10705:@&
swin_transformer_1_10707:@+
swin_transformer_1_10709:	@?'
swin_transformer_1_10711:	?+
swin_transformer_1_10713:	?@&
swin_transformer_1_10715:@'
patch_merging_10718:
??!
dense_10_10722:	?W
dense_10_10724:W
identity?? dense_10/StatefulPartitionedCall?'patch_embedding/StatefulPartitionedCall?%patch_merging/StatefulPartitionedCall?#random_crop/StatefulPartitionedCall?#random_flip/StatefulPartitionedCall?(swin_transformer/StatefulPartitionedCall?*swin_transformer_1/StatefulPartitionedCall?
#random_crop/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_crop_10576*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10491?
#random_flip/StatefulPartitionedCallStatefulPartitionedCall,random_crop/StatefulPartitionedCall:output:0random_flip_10579*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10305?
patch_extract/PartitionedCallPartitionedCall,random_flip/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9773?
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_10583patch_embedding_10585patch_embedding_10587*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9785?
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_10623swin_transformer_10625swin_transformer_10627swin_transformer_10629swin_transformer_10631swin_transformer_10633swin_transformer_10635swin_transformer_10637swin_transformer_10639swin_transformer_10641swin_transformer_10643swin_transformer_10645swin_transformer_10647swin_transformer_10649*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *1
f,R*
(__inference_restored_function_body_10622?
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_10687swin_transformer_1_10689swin_transformer_1_10691swin_transformer_1_10693swin_transformer_1_10695swin_transformer_1_10697swin_transformer_1_10699swin_transformer_1_10701swin_transformer_1_10703swin_transformer_1_10705swin_transformer_1_10707swin_transformer_1_10709swin_transformer_1_10711swin_transformer_1_10713swin_transformer_1_10715*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *1
f,R*
(__inference_restored_function_body_10686?
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_10718*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9927?
(global_average_pooling1d/PartitionedCallPartitionedCall.patch_merging/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_10_10722dense_10_10724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_10098x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????W?
NoOpNoOp!^dense_10/StatefulPartitionedCall(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall$^random_crop/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2J
#random_crop/StatefulPartitionedCall#random_crop/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_13358

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
'__inference_dense_4_layer_call_fn_13164

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_12219t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_dense_4_layer_call_and_return_conditional_losses_12219

inputs4
!tensordot_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:????????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_12545

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q???i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_13111

inputs4
!tensordot_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????e
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:???????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_12578

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q???j
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????u
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????o
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:???????????_
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_restored_function_body_9889
x
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	
	unknown_4:	 
	unknown_5:?
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:	@?

unknown_11:	?

unknown_12:	?@

unknown_13:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_2354t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????@: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
?
,__inference_sequential_1_layer_call_fn_12653
dense_7_input
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_12629t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:??????????@
'
_user_specified_namedense_7_input
?V
?
map_while_body_12032$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0K
=map_while_stateful_uniform_full_int_rngreadandskip_resource_0:	
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorI
;map_while_stateful_uniform_full_int_rngreadandskip_resource:	??2map/while/stateful_uniform_full_int/RngReadAndSkip?
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      ?
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*"
_output_shapes
:@@*
element_dtype0s
)map/while/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:s
)map/while/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
(map/while/stateful_uniform_full_int/ProdProd2map/while/stateful_uniform_full_int/shape:output:02map/while/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: l
*map/while/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
*map/while/stateful_uniform_full_int/Cast_1Cast1map/while/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
2map/while/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=map_while_stateful_uniform_full_int_rngreadandskip_resource_03map/while/stateful_uniform_full_int/Cast/x:output:0.map/while/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:?
7map/while/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9map/while/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9map/while/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1map/while/stateful_uniform_full_int/strided_sliceStridedSlice:map/while/stateful_uniform_full_int/RngReadAndSkip:value:0@map/while/stateful_uniform_full_int/strided_slice/stack:output:0Bmap/while/stateful_uniform_full_int/strided_slice/stack_1:output:0Bmap/while/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
+map/while/stateful_uniform_full_int/BitcastBitcast:map/while/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
9map/while/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
;map/while/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;map/while/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3map/while/stateful_uniform_full_int/strided_slice_1StridedSlice:map/while/stateful_uniform_full_int/RngReadAndSkip:value:0Bmap/while/stateful_uniform_full_int/strided_slice_1/stack:output:0Dmap/while/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Dmap/while/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
-map/while/stateful_uniform_full_int/Bitcast_1Bitcast<map/while/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'map/while/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
#map/while/stateful_uniform_full_intStatelessRandomUniformFullIntV22map/while/stateful_uniform_full_int/shape:output:06map/while/stateful_uniform_full_int/Bitcast_1:output:04map/while/stateful_uniform_full_int/Bitcast:output:00map/while/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	^
map/while/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ?
map/while/stackPack,map/while/stateful_uniform_full_int:output:0map/while/zeros_like:output:0*
N*
T0	*
_output_shapes

:n
map/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
map/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
map/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
map/while/strided_sliceStridedSlicemap/while/stack:output:0&map/while/strided_slice/stack:output:0(map/while/strided_slice/stack_1:output:0(map/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask?
=map/while/stateless_random_flip_left_right/control_dependencyIdentity4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*@
_class6
42loc:@map/while/TensorArrayV2Read/TensorListGetItem*"
_output_shapes
:@@?
Imap/while/stateless_random_flip_left_right/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
`map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter map/while/strided_slice:output:0* 
_output_shapes
::?
`map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
\map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Rmap/while/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0fmap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0jmap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0imap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ?
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/subSubPmap/while/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Pmap/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/mulMulemap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Kmap/while/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ?
Cmap/while/stateless_random_flip_left_right/stateless_random_uniformAddV2Kmap/while/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Pmap/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: v
1map/while/stateless_random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
/map/while/stateless_random_flip_left_right/LessLessGmap/while/stateless_random_flip_left_right/stateless_random_uniform:z:0:map/while/stateless_random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: ?
*map/while/stateless_random_flip_left_rightStatelessIf3map/while/stateless_random_flip_left_right/Less:z:0Fmap/while/stateless_random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*"
_output_shapes
:@@* 
_read_only_resource_inputs
 *I
else_branch:R8
6map_while_stateless_random_flip_left_right_false_12092*!
output_shapes
:@@*H
then_branch9R7
5map_while_stateless_random_flip_left_right_true_12091?
3map/while/stateless_random_flip_left_right/IdentityIdentity3map/while/stateless_random_flip_left_right:output:0*
T0*"
_output_shapes
:@@?
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder<map/while/stateless_random_flip_left_right/Identity:output:0*
_output_shapes
: *
element_dtype0:???Q
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: e
map/while/IdentityIdentitymap/while/add_1:z:0^map/while/NoOp*
T0*
_output_shapes
: o
map/while/Identity_1Identitymap_while_map_strided_slice^map/while/NoOp*
T0*
_output_shapes
: e
map/while/Identity_2Identitymap/while/add:z:0^map/while/NoOp*
T0*
_output_shapes
: ?
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: ?
map/while/NoOpNoOp3^map/while/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"|
;map_while_stateful_uniform_full_int_rngreadandskip_resource=map_while_stateful_uniform_full_int_rngreadandskip_resource_0"?
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2h
2map/while/stateful_uniform_full_int/RngReadAndSkip2map/while/stateful_uniform_full_int/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
c
G__inference_activation_1_layer_call_and_return_conditional_losses_13277

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*-
_output_shapes
:???????????P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???m
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*-
_output_shapes
:???????????Y
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:???????????O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:???????????e

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:???????????\
IdentityIdentityGelu/mul_1:z:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_restored_function_body_9825
x
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	
	unknown_4:	
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:	@?

unknown_10:	?

unknown_11:	?@

unknown_12:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_swin_transformer_layer_call_and_return_conditional_losses_721t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
c
G__inference_patch_extract_layer_call_and_return_conditional_losses_1291

images
identity;
ShapeShapeimages*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
ExtractImagePatchesExtractImagePatchesimages*
T0*/
_output_shapes
:?????????  *
ksizes
*
paddingVALID*
rates
*
strides
R
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*,
_output_shapes
:??????????]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameimages
?
?
F__inference_random_flip_layer_call_and_return_conditional_losses_12125

inputs
map_while_input_6:	
identity??	map/while?
	map/ShapeShapeinputs*
T0*
_output_shapes
:a
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      ?
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputsBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???K
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( * 
bodyR
map_while_body_12032* 
condR
map_while_cond_12031*!
output_shapes
: : : : : : : ?
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      ?
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*/
_output_shapes
:?????????@@*
element_dtype0?
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0^NoOp*
T0*/
_output_shapes
:?????????@@R
NoOpNoOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: 2
	map/while	map/while:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_12754

inputs
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12348t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12687
dense_7_input 
dense_7_12673:	@?
dense_7_12675:	? 
dense_8_12680:	?@
dense_8_12682:@
identity??dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCalldense_7_inputdense_7_12673dense_7_12675*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_12443?
activation_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_12461?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12578?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_8_12680dense_8_12682*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_12500?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12545~
IdentityIdentity*dropout_5/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:[ W
,
_output_shapes
:??????????@
'
_user_specified_namedense_7_input
?
?
F__inference_random_crop_layer_call_and_return_conditional_losses_10491

inputs
cond_input_1:	
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0*
value	B :@S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :@Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: g
Rank/packedPackGreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:e
	All/inputPackGreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0inputscond_input_1*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_10346*.
output_shapes
:?????????@@*"
then_branchR
cond_true_10345b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:?????????@@m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:?????????@@M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: 2
condcond:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_1156
xI
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@O
<window_attention_1_dense_5_tensordot_readvariableop_resource:	@?I
:window_attention_1_dense_5_biasadd_readvariableop_resource:	?F
4window_attention_1_reshape_1_readvariableop_resource:	4
"window_attention_1_gather_resource:	N
7window_attention_1_expanddims_1_readvariableop_resource:?N
<window_attention_1_dense_6_tensordot_readvariableop_resource:@@H
:window_attention_1_dense_6_biasadd_readvariableop_resource:@I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_3_batchnorm_readvariableop_resource:@I
6sequential_1_dense_7_tensordot_readvariableop_resource:	@?C
4sequential_1_dense_7_biasadd_readvariableop_resource:	?I
6sequential_1_dense_8_tensordot_readvariableop_resource:	?@B
4sequential_1_dense_8_biasadd_readvariableop_resource:@
identity??.layer_normalization_2/batchnorm/ReadVariableOp?2layer_normalization_2/batchnorm/mul/ReadVariableOp?.layer_normalization_3/batchnorm/ReadVariableOp?2layer_normalization_3/batchnorm/mul/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?-sequential_1/dense_7/Tensordot/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?-sequential_1/dense_8/Tensordot/ReadVariableOp?.window_attention_1/ExpandDims_1/ReadVariableOp?window_attention_1/Gather?+window_attention_1/Reshape_1/ReadVariableOp?1window_attention_1/dense_5/BiasAdd/ReadVariableOp?3window_attention_1/dense_5/Tensordot/ReadVariableOp?1window_attention_1/dense_6/BiasAdd/ReadVariableOp?3window_attention_1/dense_6/Tensordot/ReadVariableOp~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_2/moments/meanMeanx=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencex3layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/mul_1Mulx'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   ?
ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:?????????  @[

Roll/shiftConst*
_output_shapes
:*
dtype0*
valueB"????????Z
	Roll/axisConst*
_output_shapes
:*
dtype0*
valueB"      ?
RollRollReshape:output:0Roll/shift:output:0Roll/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:?????????  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   
	Reshape_1ReshapeRoll:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!?????????@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
3window_attention_1/dense_5/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0s
)window_attention_1/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)window_attention_1/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
*window_attention_1/dense_5/Tensordot/ShapeShapeReshape_3:output:0*
T0*
_output_shapes
:t
2window_attention_1/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_5/Tensordot/GatherV2GatherV23window_attention_1/dense_5/Tensordot/Shape:output:02window_attention_1/dense_5/Tensordot/free:output:0;window_attention_1/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4window_attention_1/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/window_attention_1/dense_5/Tensordot/GatherV2_1GatherV23window_attention_1/dense_5/Tensordot/Shape:output:02window_attention_1/dense_5/Tensordot/axes:output:0=window_attention_1/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*window_attention_1/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention_1/dense_5/Tensordot/ProdProd6window_attention_1/dense_5/Tensordot/GatherV2:output:03window_attention_1/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
+window_attention_1/dense_5/Tensordot/Prod_1Prod8window_attention_1/dense_5/Tensordot/GatherV2_1:output:05window_attention_1/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention_1/dense_5/Tensordot/concatConcatV22window_attention_1/dense_5/Tensordot/free:output:02window_attention_1/dense_5/Tensordot/axes:output:09window_attention_1/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*window_attention_1/dense_5/Tensordot/stackPack2window_attention_1/dense_5/Tensordot/Prod:output:04window_attention_1/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
.window_attention_1/dense_5/Tensordot/transpose	TransposeReshape_3:output:04window_attention_1/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
,window_attention_1/dense_5/Tensordot/ReshapeReshape2window_attention_1/dense_5/Tensordot/transpose:y:03window_attention_1/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
+window_attention_1/dense_5/Tensordot/MatMulMatMul5window_attention_1/dense_5/Tensordot/Reshape:output:0;window_attention_1/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
,window_attention_1/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?t
2window_attention_1/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_5/Tensordot/concat_1ConcatV26window_attention_1/dense_5/Tensordot/GatherV2:output:05window_attention_1/dense_5/Tensordot/Const_2:output:0;window_attention_1/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
$window_attention_1/dense_5/TensordotReshape5window_attention_1/dense_5/Tensordot/MatMul:product:06window_attention_1/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:???????????
1window_attention_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"window_attention_1/dense_5/BiasAddBiasAdd-window_attention_1/dense_5/Tensordot:output:09window_attention_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????}
 window_attention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention_1/ReshapeReshape+window_attention_1/dense_5/BiasAdd:output:0)window_attention_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:?????????~
!window_attention_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
window_attention_1/transpose	Transpose#window_attention_1/Reshape:output:0*window_attention_1/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????p
&window_attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(window_attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention_1/strided_sliceStridedSlice window_attention_1/transpose:y:0/window_attention_1/strided_slice/stack:output:01window_attention_1/strided_slice/stack_1:output:01window_attention_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskr
(window_attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"window_attention_1/strided_slice_1StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_1/stack:output:03window_attention_1/strided_slice_1/stack_1:output:03window_attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskr
(window_attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"window_attention_1/strided_slice_2StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_2/stack:output:03window_attention_1/strided_slice_2/stack_1:output:03window_attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_mask]
window_attention_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
window_attention_1/mulMul)window_attention_1/strided_slice:output:0!window_attention_1/mul/y:output:0*
T0*/
_output_shapes
:?????????|
#window_attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention_1/transpose_1	Transpose+window_attention_1/strided_slice_1:output:0,window_attention_1/transpose_1/perm:output:0*
T0*/
_output_shapes
:??????????
window_attention_1/matmulBatchMatMulV2window_attention_1/mul:z:0"window_attention_1/transpose_1:y:0*
T0*/
_output_shapes
:??????????
+window_attention_1/Reshape_1/ReadVariableOpReadVariableOp4window_attention_1_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	u
"window_attention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
window_attention_1/Reshape_1Reshape3window_attention_1/Reshape_1/ReadVariableOp:value:0+window_attention_1/Reshape_1/shape:output:0*
T0	*
_output_shapes
:?
window_attention_1/GatherResourceGather"window_attention_1_gather_resource%window_attention_1/Reshape_1:output:0*
Tindices0	*
_output_shapes

:*
dtype0t
window_attention_1/IdentityIdentity"window_attention_1/Gather:output:0*
T0*
_output_shapes

:w
"window_attention_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?????
window_attention_1/Reshape_2Reshape$window_attention_1/Identity:output:0+window_attention_1/Reshape_2/shape:output:0*
T0*"
_output_shapes
:x
#window_attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
window_attention_1/transpose_2	Transpose%window_attention_1/Reshape_2:output:0,window_attention_1/transpose_2/perm:output:0*
T0*"
_output_shapes
:c
!window_attention_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention_1/ExpandDims
ExpandDims"window_attention_1/transpose_2:y:0*window_attention_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:?
window_attention_1/addAddV2"window_attention_1/matmul:output:0&window_attention_1/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
.window_attention_1/ExpandDims_1/ReadVariableOpReadVariableOp7window_attention_1_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype0e
#window_attention_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
window_attention_1/ExpandDims_1
ExpandDims6window_attention_1/ExpandDims_1/ReadVariableOp:value:0,window_attention_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?e
#window_attention_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention_1/ExpandDims_2
ExpandDims(window_attention_1/ExpandDims_1:output:0,window_attention_1/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:??
window_attention_1/CastCast(window_attention_1/ExpandDims_2:output:0*

DstT0*

SrcT0*+
_output_shapes
:?
"window_attention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention_1/Reshape_3Reshapewindow_attention_1/add:z:0+window_attention_1/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????
window_attention_1/add_1AddV2%window_attention_1/Reshape_3:output:0window_attention_1/Cast:y:0*
T0*4
_output_shapes"
 :??????????{
"window_attention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
window_attention_1/Reshape_4Reshapewindow_attention_1/add_1:z:0+window_attention_1/Reshape_4/shape:output:0*
T0*/
_output_shapes
:??????????
window_attention_1/SoftmaxSoftmax%window_attention_1/Reshape_4:output:0*
T0*/
_output_shapes
:??????????
%window_attention_1/dropout_3/IdentityIdentity$window_attention_1/Softmax:softmax:0*
T0*/
_output_shapes
:??????????
window_attention_1/matmul_1BatchMatMulV2.window_attention_1/dropout_3/Identity:output:0+window_attention_1/strided_slice_2:output:0*
T0*/
_output_shapes
:?????????|
#window_attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention_1/transpose_3	Transpose$window_attention_1/matmul_1:output:0,window_attention_1/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????w
"window_attention_1/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   ?
window_attention_1/Reshape_5Reshape"window_attention_1/transpose_3:y:0+window_attention_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????@?
3window_attention_1/dense_6/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_6_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0s
)window_attention_1/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)window_attention_1/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
*window_attention_1/dense_6/Tensordot/ShapeShape%window_attention_1/Reshape_5:output:0*
T0*
_output_shapes
:t
2window_attention_1/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_6/Tensordot/GatherV2GatherV23window_attention_1/dense_6/Tensordot/Shape:output:02window_attention_1/dense_6/Tensordot/free:output:0;window_attention_1/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4window_attention_1/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/window_attention_1/dense_6/Tensordot/GatherV2_1GatherV23window_attention_1/dense_6/Tensordot/Shape:output:02window_attention_1/dense_6/Tensordot/axes:output:0=window_attention_1/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*window_attention_1/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention_1/dense_6/Tensordot/ProdProd6window_attention_1/dense_6/Tensordot/GatherV2:output:03window_attention_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
+window_attention_1/dense_6/Tensordot/Prod_1Prod8window_attention_1/dense_6/Tensordot/GatherV2_1:output:05window_attention_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention_1/dense_6/Tensordot/concatConcatV22window_attention_1/dense_6/Tensordot/free:output:02window_attention_1/dense_6/Tensordot/axes:output:09window_attention_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*window_attention_1/dense_6/Tensordot/stackPack2window_attention_1/dense_6/Tensordot/Prod:output:04window_attention_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
.window_attention_1/dense_6/Tensordot/transpose	Transpose%window_attention_1/Reshape_5:output:04window_attention_1/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
,window_attention_1/dense_6/Tensordot/ReshapeReshape2window_attention_1/dense_6/Tensordot/transpose:y:03window_attention_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
+window_attention_1/dense_6/Tensordot/MatMulMatMul5window_attention_1/dense_6/Tensordot/Reshape:output:0;window_attention_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
,window_attention_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2window_attention_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_6/Tensordot/concat_1ConcatV26window_attention_1/dense_6/Tensordot/GatherV2:output:05window_attention_1/dense_6/Tensordot/Const_2:output:0;window_attention_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
$window_attention_1/dense_6/TensordotReshape5window_attention_1/dense_6/Tensordot/MatMul:product:06window_attention_1/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"window_attention_1/dense_6/BiasAddBiasAdd-window_attention_1/dense_6/Tensordot:output:09window_attention_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@?
'window_attention_1/dropout_3/Identity_1Identity+window_attention_1/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   ?
	Reshape_4Reshape0window_attention_1/dropout_3/Identity_1:output:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:?????????@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!?????????@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:?????????  @]
Roll_1/shiftConst*
_output_shapes
:*
dtype0*
valueB"      \
Roll_1/axisConst*
_output_shapes
:*
dtype0*
valueB"      ?
Roll_1RollReshape_6:output:0Roll_1/shift:output:0Roll_1/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:?????????  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   v
	Reshape_7ReshapeRoll_1:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:??????????@S
drop_path_1/ShapeShapeReshape_7:output:0*
T0*
_output_shapes
:i
drop_path_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!drop_path_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!drop_path_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path_1/strided_sliceStridedSlicedrop_path_1/Shape:output:0(drop_path_1/strided_slice/stack:output:0*drop_path_1/strided_slice/stack_1:output:0*drop_path_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"drop_path_1/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d
"drop_path_1/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
 drop_path_1/random_uniform/shapePack"drop_path_1/strided_slice:output:0+drop_path_1/random_uniform/shape/1:output:0+drop_path_1/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:?
(drop_path_1/random_uniform/RandomUniformRandomUniform)drop_path_1/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0V
drop_path_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/addAddV2drop_path_1/add/x:output:01drop_path_1/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:?????????e
drop_path_1/FloorFloordrop_path_1/add:z:0*
T0*+
_output_shapes
:?????????Z
drop_path_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/truedivRealDivReshape_7:output:0drop_path_1/truediv/y:output:0*
T0*,
_output_shapes
:??????????@}
drop_path_1/mulMuldrop_path_1/truediv:z:0drop_path_1/Floor:y:0*
T0*,
_output_shapes
:??????????@[
addAddV2xdrop_path_1/mul:z:0*
T0*,
_output_shapes
:??????????@~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_3/moments/meanMeanadd:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/mul_1Muladd:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@?
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0m
#sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential_1/dense_7/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_7/Tensordot/GatherV2GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/free:output:05sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/dense_7/Tensordot/GatherV2_1GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/axes:output:07sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????i
$sequential_1/activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"sequential_1/activation_1/Gelu/mulMul-sequential_1/activation_1/Gelu/mul/x:output:0%sequential_1/dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:???????????j
%sequential_1/activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
&sequential_1/activation_1/Gelu/truedivRealDiv%sequential_1/dense_7/BiasAdd:output:0.sequential_1/activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:????????????
"sequential_1/activation_1/Gelu/ErfErf*sequential_1/activation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????i
$sequential_1/activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"sequential_1/activation_1/Gelu/addAddV2-sequential_1/activation_1/Gelu/add/x:output:0&sequential_1/activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
$sequential_1/activation_1/Gelu/mul_1Mul&sequential_1/activation_1/Gelu/mul:z:0&sequential_1/activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:????????????
sequential_1/dropout_4/IdentityIdentity(sequential_1/activation_1/Gelu/mul_1:z:0*
T0*-
_output_shapes
:????????????
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0m
#sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
$sequential_1/dense_8/Tensordot/ShapeShape(sequential_1/dropout_4/Identity:output:0*
T0*
_output_shapes
:n
,sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_8/Tensordot/GatherV2GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/free:output:05sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/dense_8/Tensordot/GatherV2_1GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/axes:output:07sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
(sequential_1/dense_8/Tensordot/transpose	Transpose(sequential_1/dropout_4/Identity:output:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
sequential_1/dropout_5/IdentityIdentity%sequential_1/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@k
drop_path_1/Shape_1Shape(sequential_1/dropout_5/Identity:output:0*
T0*
_output_shapes
:k
!drop_path_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#drop_path_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#drop_path_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path_1/strided_slice_1StridedSlicedrop_path_1/Shape_1:output:0*drop_path_1/strided_slice_1/stack:output:0,drop_path_1/strided_slice_1/stack_1:output:0,drop_path_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$drop_path_1/random_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :f
$drop_path_1/random_uniform_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
"drop_path_1/random_uniform_1/shapePack$drop_path_1/strided_slice_1:output:0-drop_path_1/random_uniform_1/shape/1:output:0-drop_path_1/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
*drop_path_1/random_uniform_1/RandomUniformRandomUniform+drop_path_1/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0X
drop_path_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/add_1AddV2drop_path_1/add_1/x:output:03drop_path_1/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:?????????i
drop_path_1/Floor_1Floordrop_path_1/add_1:z:0*
T0*+
_output_shapes
:?????????\
drop_path_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/truediv_1RealDiv(sequential_1/dropout_5/Identity:output:0 drop_path_1/truediv_1/y:output:0*
T0*,
_output_shapes
:??????????@?
drop_path_1/mul_1Muldrop_path_1/truediv_1:z:0drop_path_1/Floor_1:y:0*
T0*,
_output_shapes
:??????????@e
add_1AddV2add:z:0drop_path_1/mul_1:z:0*
T0*,
_output_shapes
:??????????@?
NoOpNoOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp/^window_attention_1/ExpandDims_1/ReadVariableOp^window_attention_1/Gather,^window_attention_1/Reshape_1/ReadVariableOp2^window_attention_1/dense_5/BiasAdd/ReadVariableOp4^window_attention_1/dense_5/Tensordot/ReadVariableOp2^window_attention_1/dense_6/BiasAdd/ReadVariableOp4^window_attention_1/dense_6/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????@: : : : : : : : : : : : : : : 2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2^
-sequential_1/dense_7/Tensordot/ReadVariableOp-sequential_1/dense_7/Tensordot/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2^
-sequential_1/dense_8/Tensordot/ReadVariableOp-sequential_1/dense_8/Tensordot/ReadVariableOp2`
.window_attention_1/ExpandDims_1/ReadVariableOp.window_attention_1/ExpandDims_1/ReadVariableOp26
window_attention_1/Gatherwindow_attention_1/Gather2Z
+window_attention_1/Reshape_1/ReadVariableOp+window_attention_1/Reshape_1/ReadVariableOp2f
1window_attention_1/dense_5/BiasAdd/ReadVariableOp1window_attention_1/dense_5/BiasAdd/ReadVariableOp2j
3window_attention_1/dense_5/Tensordot/ReadVariableOp3window_attention_1/dense_5/Tensordot/ReadVariableOp2f
1window_attention_1/dense_6/BiasAdd/ReadVariableOp1window_attention_1/dense_6/BiasAdd/ReadVariableOp2j
3window_attention_1/dense_6/Tensordot/ReadVariableOp3window_attention_1/dense_6/Tensordot/ReadVariableOp:O K
,
_output_shapes
:??????????@

_user_specified_namex
??
?
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_1603
xI
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@O
<window_attention_1_dense_5_tensordot_readvariableop_resource:	@?I
:window_attention_1_dense_5_biasadd_readvariableop_resource:	?F
4window_attention_1_reshape_1_readvariableop_resource:	4
"window_attention_1_gather_resource:	N
7window_attention_1_expanddims_1_readvariableop_resource:?N
<window_attention_1_dense_6_tensordot_readvariableop_resource:@@H
:window_attention_1_dense_6_biasadd_readvariableop_resource:@I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_3_batchnorm_readvariableop_resource:@I
6sequential_1_dense_7_tensordot_readvariableop_resource:	@?C
4sequential_1_dense_7_biasadd_readvariableop_resource:	?I
6sequential_1_dense_8_tensordot_readvariableop_resource:	?@B
4sequential_1_dense_8_biasadd_readvariableop_resource:@
identity??.layer_normalization_2/batchnorm/ReadVariableOp?2layer_normalization_2/batchnorm/mul/ReadVariableOp?.layer_normalization_3/batchnorm/ReadVariableOp?2layer_normalization_3/batchnorm/mul/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?-sequential_1/dense_7/Tensordot/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?-sequential_1/dense_8/Tensordot/ReadVariableOp?.window_attention_1/ExpandDims_1/ReadVariableOp?window_attention_1/Gather?+window_attention_1/Reshape_1/ReadVariableOp?1window_attention_1/dense_5/BiasAdd/ReadVariableOp?3window_attention_1/dense_5/Tensordot/ReadVariableOp?1window_attention_1/dense_6/BiasAdd/ReadVariableOp?3window_attention_1/dense_6/Tensordot/ReadVariableOp~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_2/moments/meanMeanx=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencex3layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/mul_1Mulx'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   ?
ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:?????????  @[

Roll/shiftConst*
_output_shapes
:*
dtype0*
valueB"????????Z
	Roll/axisConst*
_output_shapes
:*
dtype0*
valueB"      ?
RollRollReshape:output:0Roll/shift:output:0Roll/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:?????????  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   
	Reshape_1ReshapeRoll:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!?????????@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
3window_attention_1/dense_5/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0s
)window_attention_1/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)window_attention_1/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
*window_attention_1/dense_5/Tensordot/ShapeShapeReshape_3:output:0*
T0*
_output_shapes
:t
2window_attention_1/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_5/Tensordot/GatherV2GatherV23window_attention_1/dense_5/Tensordot/Shape:output:02window_attention_1/dense_5/Tensordot/free:output:0;window_attention_1/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4window_attention_1/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/window_attention_1/dense_5/Tensordot/GatherV2_1GatherV23window_attention_1/dense_5/Tensordot/Shape:output:02window_attention_1/dense_5/Tensordot/axes:output:0=window_attention_1/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*window_attention_1/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention_1/dense_5/Tensordot/ProdProd6window_attention_1/dense_5/Tensordot/GatherV2:output:03window_attention_1/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
+window_attention_1/dense_5/Tensordot/Prod_1Prod8window_attention_1/dense_5/Tensordot/GatherV2_1:output:05window_attention_1/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention_1/dense_5/Tensordot/concatConcatV22window_attention_1/dense_5/Tensordot/free:output:02window_attention_1/dense_5/Tensordot/axes:output:09window_attention_1/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*window_attention_1/dense_5/Tensordot/stackPack2window_attention_1/dense_5/Tensordot/Prod:output:04window_attention_1/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
.window_attention_1/dense_5/Tensordot/transpose	TransposeReshape_3:output:04window_attention_1/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
,window_attention_1/dense_5/Tensordot/ReshapeReshape2window_attention_1/dense_5/Tensordot/transpose:y:03window_attention_1/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
+window_attention_1/dense_5/Tensordot/MatMulMatMul5window_attention_1/dense_5/Tensordot/Reshape:output:0;window_attention_1/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
,window_attention_1/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?t
2window_attention_1/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_5/Tensordot/concat_1ConcatV26window_attention_1/dense_5/Tensordot/GatherV2:output:05window_attention_1/dense_5/Tensordot/Const_2:output:0;window_attention_1/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
$window_attention_1/dense_5/TensordotReshape5window_attention_1/dense_5/Tensordot/MatMul:product:06window_attention_1/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:???????????
1window_attention_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"window_attention_1/dense_5/BiasAddBiasAdd-window_attention_1/dense_5/Tensordot:output:09window_attention_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????}
 window_attention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention_1/ReshapeReshape+window_attention_1/dense_5/BiasAdd:output:0)window_attention_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:?????????~
!window_attention_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
window_attention_1/transpose	Transpose#window_attention_1/Reshape:output:0*window_attention_1/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????p
&window_attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(window_attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention_1/strided_sliceStridedSlice window_attention_1/transpose:y:0/window_attention_1/strided_slice/stack:output:01window_attention_1/strided_slice/stack_1:output:01window_attention_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskr
(window_attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"window_attention_1/strided_slice_1StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_1/stack:output:03window_attention_1/strided_slice_1/stack_1:output:03window_attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskr
(window_attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"window_attention_1/strided_slice_2StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_2/stack:output:03window_attention_1/strided_slice_2/stack_1:output:03window_attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_mask]
window_attention_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
window_attention_1/mulMul)window_attention_1/strided_slice:output:0!window_attention_1/mul/y:output:0*
T0*/
_output_shapes
:?????????|
#window_attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention_1/transpose_1	Transpose+window_attention_1/strided_slice_1:output:0,window_attention_1/transpose_1/perm:output:0*
T0*/
_output_shapes
:??????????
window_attention_1/matmulBatchMatMulV2window_attention_1/mul:z:0"window_attention_1/transpose_1:y:0*
T0*/
_output_shapes
:??????????
+window_attention_1/Reshape_1/ReadVariableOpReadVariableOp4window_attention_1_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	u
"window_attention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
window_attention_1/Reshape_1Reshape3window_attention_1/Reshape_1/ReadVariableOp:value:0+window_attention_1/Reshape_1/shape:output:0*
T0	*
_output_shapes
:?
window_attention_1/GatherResourceGather"window_attention_1_gather_resource%window_attention_1/Reshape_1:output:0*
Tindices0	*
_output_shapes

:*
dtype0t
window_attention_1/IdentityIdentity"window_attention_1/Gather:output:0*
T0*
_output_shapes

:w
"window_attention_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?????
window_attention_1/Reshape_2Reshape$window_attention_1/Identity:output:0+window_attention_1/Reshape_2/shape:output:0*
T0*"
_output_shapes
:x
#window_attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
window_attention_1/transpose_2	Transpose%window_attention_1/Reshape_2:output:0,window_attention_1/transpose_2/perm:output:0*
T0*"
_output_shapes
:c
!window_attention_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention_1/ExpandDims
ExpandDims"window_attention_1/transpose_2:y:0*window_attention_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:?
window_attention_1/addAddV2"window_attention_1/matmul:output:0&window_attention_1/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
.window_attention_1/ExpandDims_1/ReadVariableOpReadVariableOp7window_attention_1_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype0e
#window_attention_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
window_attention_1/ExpandDims_1
ExpandDims6window_attention_1/ExpandDims_1/ReadVariableOp:value:0,window_attention_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?e
#window_attention_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention_1/ExpandDims_2
ExpandDims(window_attention_1/ExpandDims_1:output:0,window_attention_1/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:??
window_attention_1/CastCast(window_attention_1/ExpandDims_2:output:0*

DstT0*

SrcT0*+
_output_shapes
:?
"window_attention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention_1/Reshape_3Reshapewindow_attention_1/add:z:0+window_attention_1/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????
window_attention_1/add_1AddV2%window_attention_1/Reshape_3:output:0window_attention_1/Cast:y:0*
T0*4
_output_shapes"
 :??????????{
"window_attention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
window_attention_1/Reshape_4Reshapewindow_attention_1/add_1:z:0+window_attention_1/Reshape_4/shape:output:0*
T0*/
_output_shapes
:??????????
window_attention_1/SoftmaxSoftmax%window_attention_1/Reshape_4:output:0*
T0*/
_output_shapes
:?????????o
*window_attention_1/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
(window_attention_1/dropout_3/dropout/MulMul$window_attention_1/Softmax:softmax:03window_attention_1/dropout_3/dropout/Const:output:0*
T0*/
_output_shapes
:?????????~
*window_attention_1/dropout_3/dropout/ShapeShape$window_attention_1/Softmax:softmax:0*
T0*
_output_shapes
:?
Awindow_attention_1/dropout_3/dropout/random_uniform/RandomUniformRandomUniform3window_attention_1/dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0x
3window_attention_1/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
1window_attention_1/dropout_3/dropout/GreaterEqualGreaterEqualJwindow_attention_1/dropout_3/dropout/random_uniform/RandomUniform:output:0<window_attention_1/dropout_3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
)window_attention_1/dropout_3/dropout/CastCast5window_attention_1/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
*window_attention_1/dropout_3/dropout/Mul_1Mul,window_attention_1/dropout_3/dropout/Mul:z:0-window_attention_1/dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
window_attention_1/matmul_1BatchMatMulV2.window_attention_1/dropout_3/dropout/Mul_1:z:0+window_attention_1/strided_slice_2:output:0*
T0*/
_output_shapes
:?????????|
#window_attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention_1/transpose_3	Transpose$window_attention_1/matmul_1:output:0,window_attention_1/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????w
"window_attention_1/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   ?
window_attention_1/Reshape_5Reshape"window_attention_1/transpose_3:y:0+window_attention_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????@?
3window_attention_1/dense_6/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_6_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0s
)window_attention_1/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)window_attention_1/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
*window_attention_1/dense_6/Tensordot/ShapeShape%window_attention_1/Reshape_5:output:0*
T0*
_output_shapes
:t
2window_attention_1/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_6/Tensordot/GatherV2GatherV23window_attention_1/dense_6/Tensordot/Shape:output:02window_attention_1/dense_6/Tensordot/free:output:0;window_attention_1/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4window_attention_1/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/window_attention_1/dense_6/Tensordot/GatherV2_1GatherV23window_attention_1/dense_6/Tensordot/Shape:output:02window_attention_1/dense_6/Tensordot/axes:output:0=window_attention_1/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*window_attention_1/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention_1/dense_6/Tensordot/ProdProd6window_attention_1/dense_6/Tensordot/GatherV2:output:03window_attention_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
+window_attention_1/dense_6/Tensordot/Prod_1Prod8window_attention_1/dense_6/Tensordot/GatherV2_1:output:05window_attention_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention_1/dense_6/Tensordot/concatConcatV22window_attention_1/dense_6/Tensordot/free:output:02window_attention_1/dense_6/Tensordot/axes:output:09window_attention_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*window_attention_1/dense_6/Tensordot/stackPack2window_attention_1/dense_6/Tensordot/Prod:output:04window_attention_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
.window_attention_1/dense_6/Tensordot/transpose	Transpose%window_attention_1/Reshape_5:output:04window_attention_1/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
,window_attention_1/dense_6/Tensordot/ReshapeReshape2window_attention_1/dense_6/Tensordot/transpose:y:03window_attention_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
+window_attention_1/dense_6/Tensordot/MatMulMatMul5window_attention_1/dense_6/Tensordot/Reshape:output:0;window_attention_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
,window_attention_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2window_attention_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_6/Tensordot/concat_1ConcatV26window_attention_1/dense_6/Tensordot/GatherV2:output:05window_attention_1/dense_6/Tensordot/Const_2:output:0;window_attention_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
$window_attention_1/dense_6/TensordotReshape5window_attention_1/dense_6/Tensordot/MatMul:product:06window_attention_1/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"window_attention_1/dense_6/BiasAddBiasAdd-window_attention_1/dense_6/Tensordot:output:09window_attention_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@q
,window_attention_1/dropout_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
*window_attention_1/dropout_3/dropout_1/MulMul+window_attention_1/dense_6/BiasAdd:output:05window_attention_1/dropout_3/dropout_1/Const:output:0*
T0*+
_output_shapes
:?????????@?
,window_attention_1/dropout_3/dropout_1/ShapeShape+window_attention_1/dense_6/BiasAdd:output:0*
T0*
_output_shapes
:?
Cwindow_attention_1/dropout_3/dropout_1/random_uniform/RandomUniformRandomUniform5window_attention_1/dropout_3/dropout_1/Shape:output:0*
T0*+
_output_shapes
:?????????@*
dtype0z
5window_attention_1/dropout_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
3window_attention_1/dropout_3/dropout_1/GreaterEqualGreaterEqualLwindow_attention_1/dropout_3/dropout_1/random_uniform/RandomUniform:output:0>window_attention_1/dropout_3/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????@?
+window_attention_1/dropout_3/dropout_1/CastCast7window_attention_1/dropout_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????@?
,window_attention_1/dropout_3/dropout_1/Mul_1Mul.window_attention_1/dropout_3/dropout_1/Mul:z:0/window_attention_1/dropout_3/dropout_1/Cast:y:0*
T0*+
_output_shapes
:?????????@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   ?
	Reshape_4Reshape0window_attention_1/dropout_3/dropout_1/Mul_1:z:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:?????????@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!?????????@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:?????????  @]
Roll_1/shiftConst*
_output_shapes
:*
dtype0*
valueB"      \
Roll_1/axisConst*
_output_shapes
:*
dtype0*
valueB"      ?
Roll_1RollReshape_6:output:0Roll_1/shift:output:0Roll_1/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:?????????  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   v
	Reshape_7ReshapeRoll_1:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:??????????@S
drop_path_1/ShapeShapeReshape_7:output:0*
T0*
_output_shapes
:i
drop_path_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!drop_path_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!drop_path_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path_1/strided_sliceStridedSlicedrop_path_1/Shape:output:0(drop_path_1/strided_slice/stack:output:0*drop_path_1/strided_slice/stack_1:output:0*drop_path_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"drop_path_1/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d
"drop_path_1/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
 drop_path_1/random_uniform/shapePack"drop_path_1/strided_slice:output:0+drop_path_1/random_uniform/shape/1:output:0+drop_path_1/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:?
(drop_path_1/random_uniform/RandomUniformRandomUniform)drop_path_1/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0V
drop_path_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/addAddV2drop_path_1/add/x:output:01drop_path_1/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:?????????e
drop_path_1/FloorFloordrop_path_1/add:z:0*
T0*+
_output_shapes
:?????????Z
drop_path_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/truedivRealDivReshape_7:output:0drop_path_1/truediv/y:output:0*
T0*,
_output_shapes
:??????????@}
drop_path_1/mulMuldrop_path_1/truediv:z:0drop_path_1/Floor:y:0*
T0*,
_output_shapes
:??????????@[
addAddV2xdrop_path_1/mul:z:0*
T0*,
_output_shapes
:??????????@~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_3/moments/meanMeanadd:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/mul_1Muladd:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@?
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0m
#sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential_1/dense_7/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_7/Tensordot/GatherV2GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/free:output:05sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/dense_7/Tensordot/GatherV2_1GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/axes:output:07sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????i
$sequential_1/activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"sequential_1/activation_1/Gelu/mulMul-sequential_1/activation_1/Gelu/mul/x:output:0%sequential_1/dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:???????????j
%sequential_1/activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
&sequential_1/activation_1/Gelu/truedivRealDiv%sequential_1/dense_7/BiasAdd:output:0.sequential_1/activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:????????????
"sequential_1/activation_1/Gelu/ErfErf*sequential_1/activation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????i
$sequential_1/activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"sequential_1/activation_1/Gelu/addAddV2-sequential_1/activation_1/Gelu/add/x:output:0&sequential_1/activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
$sequential_1/activation_1/Gelu/mul_1Mul&sequential_1/activation_1/Gelu/mul:z:0&sequential_1/activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:???????????i
$sequential_1/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
"sequential_1/dropout_4/dropout/MulMul(sequential_1/activation_1/Gelu/mul_1:z:0-sequential_1/dropout_4/dropout/Const:output:0*
T0*-
_output_shapes
:???????????|
$sequential_1/dropout_4/dropout/ShapeShape(sequential_1/activation_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:?
;sequential_1/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_4/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0r
-sequential_1/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
+sequential_1/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_4/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
#sequential_1/dropout_4/dropout/CastCast/sequential_1/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
$sequential_1/dropout_4/dropout/Mul_1Mul&sequential_1/dropout_4/dropout/Mul:z:0'sequential_1/dropout_4/dropout/Cast:y:0*
T0*-
_output_shapes
:????????????
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0m
#sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
$sequential_1/dense_8/Tensordot/ShapeShape(sequential_1/dropout_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:n
,sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_8/Tensordot/GatherV2GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/free:output:05sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/dense_8/Tensordot/GatherV2_1GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/axes:output:07sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
(sequential_1/dense_8/Tensordot/transpose	Transpose(sequential_1/dropout_4/dropout/Mul_1:z:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@i
$sequential_1/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
"sequential_1/dropout_5/dropout/MulMul%sequential_1/dense_8/BiasAdd:output:0-sequential_1/dropout_5/dropout/Const:output:0*
T0*,
_output_shapes
:??????????@y
$sequential_1/dropout_5/dropout/ShapeShape%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:?
;sequential_1/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_5/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype0r
-sequential_1/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
+sequential_1/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_5/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@?
#sequential_1/dropout_5/dropout/CastCast/sequential_1/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@?
$sequential_1/dropout_5/dropout/Mul_1Mul&sequential_1/dropout_5/dropout/Mul:z:0'sequential_1/dropout_5/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@k
drop_path_1/Shape_1Shape(sequential_1/dropout_5/dropout/Mul_1:z:0*
T0*
_output_shapes
:k
!drop_path_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#drop_path_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#drop_path_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path_1/strided_slice_1StridedSlicedrop_path_1/Shape_1:output:0*drop_path_1/strided_slice_1/stack:output:0,drop_path_1/strided_slice_1/stack_1:output:0,drop_path_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$drop_path_1/random_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :f
$drop_path_1/random_uniform_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
"drop_path_1/random_uniform_1/shapePack$drop_path_1/strided_slice_1:output:0-drop_path_1/random_uniform_1/shape/1:output:0-drop_path_1/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
*drop_path_1/random_uniform_1/RandomUniformRandomUniform+drop_path_1/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0X
drop_path_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/add_1AddV2drop_path_1/add_1/x:output:03drop_path_1/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:?????????i
drop_path_1/Floor_1Floordrop_path_1/add_1:z:0*
T0*+
_output_shapes
:?????????\
drop_path_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/truediv_1RealDiv(sequential_1/dropout_5/dropout/Mul_1:z:0 drop_path_1/truediv_1/y:output:0*
T0*,
_output_shapes
:??????????@?
drop_path_1/mul_1Muldrop_path_1/truediv_1:z:0drop_path_1/Floor_1:y:0*
T0*,
_output_shapes
:??????????@e
add_1AddV2add:z:0drop_path_1/mul_1:z:0*
T0*,
_output_shapes
:??????????@?
NoOpNoOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp/^window_attention_1/ExpandDims_1/ReadVariableOp^window_attention_1/Gather,^window_attention_1/Reshape_1/ReadVariableOp2^window_attention_1/dense_5/BiasAdd/ReadVariableOp4^window_attention_1/dense_5/Tensordot/ReadVariableOp2^window_attention_1/dense_6/BiasAdd/ReadVariableOp4^window_attention_1/dense_6/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????@: : : : : : : : : : : : : : : 2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2^
-sequential_1/dense_7/Tensordot/ReadVariableOp-sequential_1/dense_7/Tensordot/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2^
-sequential_1/dense_8/Tensordot/ReadVariableOp-sequential_1/dense_8/Tensordot/ReadVariableOp2`
.window_attention_1/ExpandDims_1/ReadVariableOp.window_attention_1/ExpandDims_1/ReadVariableOp26
window_attention_1/Gatherwindow_attention_1/Gather2Z
+window_attention_1/Reshape_1/ReadVariableOp+window_attention_1/Reshape_1/ReadVariableOp2f
1window_attention_1/dense_5/BiasAdd/ReadVariableOp1window_attention_1/dense_5/BiasAdd/ReadVariableOp2j
3window_attention_1/dense_5/Tensordot/ReadVariableOp3window_attention_1/dense_5/Tensordot/ReadVariableOp2f
1window_attention_1/dense_6/BiasAdd/ReadVariableOp1window_attention_1/dense_6/BiasAdd/ReadVariableOp2j
3window_attention_1/dense_6/Tensordot/ReadVariableOp3window_attention_1/dense_6/Tensordot/ReadVariableOp:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
?
5map_while_stateless_random_flip_left_right_true_12091v
rmap_while_stateless_random_flip_left_right_reversev2_map_while_stateless_random_flip_left_right_control_dependency7
3map_while_stateless_random_flip_left_right_identity?
9map/while/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:?
4map/while/stateless_random_flip_left_right/ReverseV2	ReverseV2rmap_while_stateless_random_flip_left_right_reversev2_map_while_stateless_random_flip_left_right_control_dependencyBmap/while/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*"
_output_shapes
:@@?
3map/while/stateless_random_flip_left_right/IdentityIdentity=map/while/stateless_random_flip_left_right/ReverseV2:output:0*
T0*"
_output_shapes
:@@"s
3map_while_stateless_random_flip_left_right_identity<map/while/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:@@:( $
"
_output_shapes
:@@
?
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?/
?
random_crop_cond_false_11439!
random_crop_cond_shape_inputs 
random_crop_cond_placeholder
random_crop_cond_identityc
random_crop/cond/ShapeShaperandom_crop_cond_shape_inputs*
T0*
_output_shapes
:w
$random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????y
&random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????p
&random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/cond/strided_sliceStridedSlicerandom_crop/cond/Shape:output:0-random_crop/cond/strided_slice/stack:output:0/random_crop/cond/strided_slice/stack_1:output:0/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
&random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
(random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop/cond/strided_slice_1StridedSlicerandom_crop/cond/Shape:output:0/random_crop/cond/strided_slice_1/stack:output:01random_crop/cond/strided_slice_1/stack_1:output:01random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
random_crop/cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@?
random_crop/cond/mulMul)random_crop/cond/strided_slice_1:output:0random_crop/cond/mul/y:output:0*
T0*
_output_shapes
: g
random_crop/cond/CastCastrandom_crop/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: _
random_crop/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B?
random_crop/cond/truedivRealDivrandom_crop/cond/Cast:y:0#random_crop/cond/truediv/y:output:0*
T0*
_output_shapes
: m
random_crop/cond/Cast_1Castrandom_crop/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
random_crop/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :@?
random_crop/cond/mul_1Mul'random_crop/cond/strided_slice:output:0!random_crop/cond/mul_1/y:output:0*
T0*
_output_shapes
: k
random_crop/cond/Cast_2Castrandom_crop/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B?
random_crop/cond/truediv_1RealDivrandom_crop/cond/Cast_2:y:0%random_crop/cond/truediv_1/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_3Castrandom_crop/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
random_crop/cond/MinimumMinimum'random_crop/cond/strided_slice:output:0random_crop/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
random_crop/cond/Minimum_1Minimum)random_crop/cond/strided_slice_1:output:0random_crop/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
random_crop/cond/subSub'random_crop/cond/strided_slice:output:0random_crop/cond/Minimum:z:0*
T0*
_output_shapes
: i
random_crop/cond/Cast_4Castrandom_crop/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
random_crop/cond/truediv_2RealDivrandom_crop/cond/Cast_4:y:0%random_crop/cond/truediv_2/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_5Castrandom_crop/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
random_crop/cond/sub_1Sub)random_crop/cond/strided_slice_1:output:0random_crop/cond/Minimum_1:z:0*
T0*
_output_shapes
: k
random_crop/cond/Cast_6Castrandom_crop/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
random_crop/cond/truediv_3RealDivrandom_crop/cond/Cast_6:y:0%random_crop/cond/truediv_3/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_7Castrandom_crop/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: Z
random_crop/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : Z
random_crop/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/cond/stackPack!random_crop/cond/stack/0:output:0random_crop/cond/Cast_5:y:0random_crop/cond/Cast_7:y:0!random_crop/cond/stack/3:output:0*
N*
T0*
_output_shapes
:e
random_crop/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????e
random_crop/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
random_crop/cond/stack_1Pack#random_crop/cond/stack_1/0:output:0random_crop/cond/Minimum:z:0random_crop/cond/Minimum_1:z:0#random_crop/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
random_crop/cond/SliceSlicerandom_crop_cond_shape_inputsrandom_crop/cond/stack:output:0!random_crop/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????@@?????????m
random_crop/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   ?
&random_crop/cond/resize/ResizeBilinearResizeBilinearrandom_crop/cond/Slice:output:0%random_crop/cond/resize/size:output:0*
T0*/
_output_shapes
:?????????@@*
half_pixel_centers(?
random_crop/cond/IdentityIdentity7random_crop/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????@@"?
random_crop_cond_identity"random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: :5 1
/
_output_shapes
:?????????@@
?
b
)__inference_dropout_5_layer_call_fn_13353

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12545t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
B__inference_dense_8_layer_call_and_return_conditional_losses_13343

inputs4
!tensordot_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:????????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_12926

inputs
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_12629t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_12162

inputs4
!tensordot_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????e
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:???????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?G
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12992

inputs<
)dense_7_tensordot_readvariableop_resource:	@?6
'dense_7_biasadd_readvariableop_resource:	?<
)dense_8_tensordot_readvariableop_resource:	?@5
'dense_8_biasadd_readvariableop_resource:@
identity??dense_7/BiasAdd/ReadVariableOp? dense_7/Tensordot/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp? dense_8/Tensordot/ReadVariableOp?
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0`
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       M
dense_7/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:a
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_7/Tensordot/transpose	Transposeinputs!dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????\
activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
activation_1/Gelu/mulMul activation_1/Gelu/mul/x:output:0dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:???????????]
activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
activation_1/Gelu/truedivRealDivdense_7/BiasAdd:output:0!activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:???????????s
activation_1/Gelu/ErfErfactivation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????\
activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
activation_1/Gelu/addAddV2 activation_1/Gelu/add/x:output:0activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
activation_1/Gelu/mul_1Mulactivation_1/Gelu/mul:z:0activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:???????????s
dropout_4/IdentityIdentityactivation_1/Gelu/mul_1:z:0*
T0*-
_output_shapes
:????????????
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0`
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_8/Tensordot/ShapeShapedropout_4/Identity:output:0*
T0*
_output_shapes
:a
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_8/Tensordot/transpose	Transposedropout_4/Identity:output:0!dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@c
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@o
dropout_5/IdentityIdentitydense_8/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@o
IdentityIdentitydropout_5/Identity:output:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_12297

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q???j
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????u
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????o
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:???????????_
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?

c
D__inference_dropout_5_layer_call_and_return_conditional_losses_13370

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q???i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_12741

inputs
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12233t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?(
?
I__inference_patch_embedding_layer_call_and_return_conditional_losses_2689	
patch9
'dense_tensordot_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@4
!embedding_embedding_lookup_372620:	?@
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?embedding/embedding_lookupM
range/startConst*
_output_shapes
: *
dtype0*
value	B : N
range/limitConst*
_output_shapes
: *
dtype0*
value
B :?M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :m
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:??
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       J
dense/Tensordot/ShapeShapepatch*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transposepatchdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:???????????
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_372620range:output:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/372620*
_output_shapes
:	?@*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/372620*
_output_shapes
:	?@?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	?@?
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????@?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 [
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:S O
,
_output_shapes
:??????????

_user_specified_namepatch
?
{
+__inference_random_flip_layer_call_fn_12012

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10305w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
a
E__inference_activation_layer_call_and_return_conditional_losses_13128

inputs
identityO

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
Gelu/mulMulGelu/mul/x:output:0inputs*
T0*-
_output_shapes
:???????????P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???m
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*-
_output_shapes
:???????????Y
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:???????????O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??l
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:???????????e

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:???????????\
IdentityIdentityGelu/mul_1:z:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_6078
xI
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@O
<window_attention_1_dense_5_tensordot_readvariableop_resource:	@?I
:window_attention_1_dense_5_biasadd_readvariableop_resource:	?F
4window_attention_1_reshape_1_readvariableop_resource:	4
"window_attention_1_gather_resource:	N
7window_attention_1_expanddims_1_readvariableop_resource:?N
<window_attention_1_dense_6_tensordot_readvariableop_resource:@@H
:window_attention_1_dense_6_biasadd_readvariableop_resource:@I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_3_batchnorm_readvariableop_resource:@I
6sequential_1_dense_7_tensordot_readvariableop_resource:	@?C
4sequential_1_dense_7_biasadd_readvariableop_resource:	?I
6sequential_1_dense_8_tensordot_readvariableop_resource:	?@B
4sequential_1_dense_8_biasadd_readvariableop_resource:@
identity??.layer_normalization_2/batchnorm/ReadVariableOp?2layer_normalization_2/batchnorm/mul/ReadVariableOp?.layer_normalization_3/batchnorm/ReadVariableOp?2layer_normalization_3/batchnorm/mul/ReadVariableOp?+sequential_1/dense_7/BiasAdd/ReadVariableOp?-sequential_1/dense_7/Tensordot/ReadVariableOp?+sequential_1/dense_8/BiasAdd/ReadVariableOp?-sequential_1/dense_8/Tensordot/ReadVariableOp?.window_attention_1/ExpandDims_1/ReadVariableOp?window_attention_1/Gather?+window_attention_1/Reshape_1/ReadVariableOp?1window_attention_1/dense_5/BiasAdd/ReadVariableOp?3window_attention_1/dense_5/Tensordot/ReadVariableOp?1window_attention_1/dense_6/BiasAdd/ReadVariableOp?3window_attention_1/dense_6/Tensordot/ReadVariableOp~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_2/moments/meanMeanx=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencex3layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/mul_1Mulx'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   ?
ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:?????????  @[

Roll/shiftConst*
_output_shapes
:*
dtype0*
valueB"????????Z
	Roll/axisConst*
_output_shapes
:*
dtype0*
valueB"      ?
RollRollReshape:output:0Roll/shift:output:0Roll/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:?????????  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   
	Reshape_1ReshapeRoll:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!?????????@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
3window_attention_1/dense_5/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0s
)window_attention_1/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)window_attention_1/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
*window_attention_1/dense_5/Tensordot/ShapeShapeReshape_3:output:0*
T0*
_output_shapes
:t
2window_attention_1/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_5/Tensordot/GatherV2GatherV23window_attention_1/dense_5/Tensordot/Shape:output:02window_attention_1/dense_5/Tensordot/free:output:0;window_attention_1/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4window_attention_1/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/window_attention_1/dense_5/Tensordot/GatherV2_1GatherV23window_attention_1/dense_5/Tensordot/Shape:output:02window_attention_1/dense_5/Tensordot/axes:output:0=window_attention_1/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*window_attention_1/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention_1/dense_5/Tensordot/ProdProd6window_attention_1/dense_5/Tensordot/GatherV2:output:03window_attention_1/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
+window_attention_1/dense_5/Tensordot/Prod_1Prod8window_attention_1/dense_5/Tensordot/GatherV2_1:output:05window_attention_1/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention_1/dense_5/Tensordot/concatConcatV22window_attention_1/dense_5/Tensordot/free:output:02window_attention_1/dense_5/Tensordot/axes:output:09window_attention_1/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*window_attention_1/dense_5/Tensordot/stackPack2window_attention_1/dense_5/Tensordot/Prod:output:04window_attention_1/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
.window_attention_1/dense_5/Tensordot/transpose	TransposeReshape_3:output:04window_attention_1/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
,window_attention_1/dense_5/Tensordot/ReshapeReshape2window_attention_1/dense_5/Tensordot/transpose:y:03window_attention_1/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
+window_attention_1/dense_5/Tensordot/MatMulMatMul5window_attention_1/dense_5/Tensordot/Reshape:output:0;window_attention_1/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
,window_attention_1/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?t
2window_attention_1/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_5/Tensordot/concat_1ConcatV26window_attention_1/dense_5/Tensordot/GatherV2:output:05window_attention_1/dense_5/Tensordot/Const_2:output:0;window_attention_1/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
$window_attention_1/dense_5/TensordotReshape5window_attention_1/dense_5/Tensordot/MatMul:product:06window_attention_1/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:???????????
1window_attention_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"window_attention_1/dense_5/BiasAddBiasAdd-window_attention_1/dense_5/Tensordot:output:09window_attention_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????}
 window_attention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention_1/ReshapeReshape+window_attention_1/dense_5/BiasAdd:output:0)window_attention_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:?????????~
!window_attention_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
window_attention_1/transpose	Transpose#window_attention_1/Reshape:output:0*window_attention_1/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????p
&window_attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(window_attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention_1/strided_sliceStridedSlice window_attention_1/transpose:y:0/window_attention_1/strided_slice/stack:output:01window_attention_1/strided_slice/stack_1:output:01window_attention_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskr
(window_attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"window_attention_1/strided_slice_1StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_1/stack:output:03window_attention_1/strided_slice_1/stack_1:output:03window_attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskr
(window_attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*window_attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"window_attention_1/strided_slice_2StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_2/stack:output:03window_attention_1/strided_slice_2/stack_1:output:03window_attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_mask]
window_attention_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
window_attention_1/mulMul)window_attention_1/strided_slice:output:0!window_attention_1/mul/y:output:0*
T0*/
_output_shapes
:?????????|
#window_attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention_1/transpose_1	Transpose+window_attention_1/strided_slice_1:output:0,window_attention_1/transpose_1/perm:output:0*
T0*/
_output_shapes
:??????????
window_attention_1/matmulBatchMatMulV2window_attention_1/mul:z:0"window_attention_1/transpose_1:y:0*
T0*/
_output_shapes
:??????????
+window_attention_1/Reshape_1/ReadVariableOpReadVariableOp4window_attention_1_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	u
"window_attention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
window_attention_1/Reshape_1Reshape3window_attention_1/Reshape_1/ReadVariableOp:value:0+window_attention_1/Reshape_1/shape:output:0*
T0	*
_output_shapes
:?
window_attention_1/GatherResourceGather"window_attention_1_gather_resource%window_attention_1/Reshape_1:output:0*
Tindices0	*
_output_shapes

:*
dtype0t
window_attention_1/IdentityIdentity"window_attention_1/Gather:output:0*
T0*
_output_shapes

:w
"window_attention_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?????
window_attention_1/Reshape_2Reshape$window_attention_1/Identity:output:0+window_attention_1/Reshape_2/shape:output:0*
T0*"
_output_shapes
:x
#window_attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
window_attention_1/transpose_2	Transpose%window_attention_1/Reshape_2:output:0,window_attention_1/transpose_2/perm:output:0*
T0*"
_output_shapes
:c
!window_attention_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention_1/ExpandDims
ExpandDims"window_attention_1/transpose_2:y:0*window_attention_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:?
window_attention_1/addAddV2"window_attention_1/matmul:output:0&window_attention_1/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
.window_attention_1/ExpandDims_1/ReadVariableOpReadVariableOp7window_attention_1_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype0e
#window_attention_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :?
window_attention_1/ExpandDims_1
ExpandDims6window_attention_1/ExpandDims_1/ReadVariableOp:value:0,window_attention_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?e
#window_attention_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention_1/ExpandDims_2
ExpandDims(window_attention_1/ExpandDims_1:output:0,window_attention_1/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:??
window_attention_1/CastCast(window_attention_1/ExpandDims_2:output:0*

DstT0*

SrcT0*+
_output_shapes
:?
"window_attention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention_1/Reshape_3Reshapewindow_attention_1/add:z:0+window_attention_1/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :???????????
window_attention_1/add_1AddV2%window_attention_1/Reshape_3:output:0window_attention_1/Cast:y:0*
T0*4
_output_shapes"
 :??????????{
"window_attention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
window_attention_1/Reshape_4Reshapewindow_attention_1/add_1:z:0+window_attention_1/Reshape_4/shape:output:0*
T0*/
_output_shapes
:??????????
window_attention_1/SoftmaxSoftmax%window_attention_1/Reshape_4:output:0*
T0*/
_output_shapes
:?????????o
*window_attention_1/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
(window_attention_1/dropout_3/dropout/MulMul$window_attention_1/Softmax:softmax:03window_attention_1/dropout_3/dropout/Const:output:0*
T0*/
_output_shapes
:?????????~
*window_attention_1/dropout_3/dropout/ShapeShape$window_attention_1/Softmax:softmax:0*
T0*
_output_shapes
:?
Awindow_attention_1/dropout_3/dropout/random_uniform/RandomUniformRandomUniform3window_attention_1/dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0x
3window_attention_1/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
1window_attention_1/dropout_3/dropout/GreaterEqualGreaterEqualJwindow_attention_1/dropout_3/dropout/random_uniform/RandomUniform:output:0<window_attention_1/dropout_3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
)window_attention_1/dropout_3/dropout/CastCast5window_attention_1/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
*window_attention_1/dropout_3/dropout/Mul_1Mul,window_attention_1/dropout_3/dropout/Mul:z:0-window_attention_1/dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
window_attention_1/matmul_1BatchMatMulV2.window_attention_1/dropout_3/dropout/Mul_1:z:0+window_attention_1/strided_slice_2:output:0*
T0*/
_output_shapes
:?????????|
#window_attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention_1/transpose_3	Transpose$window_attention_1/matmul_1:output:0,window_attention_1/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????w
"window_attention_1/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   ?
window_attention_1/Reshape_5Reshape"window_attention_1/transpose_3:y:0+window_attention_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????@?
3window_attention_1/dense_6/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_6_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0s
)window_attention_1/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)window_attention_1/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
*window_attention_1/dense_6/Tensordot/ShapeShape%window_attention_1/Reshape_5:output:0*
T0*
_output_shapes
:t
2window_attention_1/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_6/Tensordot/GatherV2GatherV23window_attention_1/dense_6/Tensordot/Shape:output:02window_attention_1/dense_6/Tensordot/free:output:0;window_attention_1/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4window_attention_1/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/window_attention_1/dense_6/Tensordot/GatherV2_1GatherV23window_attention_1/dense_6/Tensordot/Shape:output:02window_attention_1/dense_6/Tensordot/axes:output:0=window_attention_1/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*window_attention_1/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention_1/dense_6/Tensordot/ProdProd6window_attention_1/dense_6/Tensordot/GatherV2:output:03window_attention_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
+window_attention_1/dense_6/Tensordot/Prod_1Prod8window_attention_1/dense_6/Tensordot/GatherV2_1:output:05window_attention_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention_1/dense_6/Tensordot/concatConcatV22window_attention_1/dense_6/Tensordot/free:output:02window_attention_1/dense_6/Tensordot/axes:output:09window_attention_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
*window_attention_1/dense_6/Tensordot/stackPack2window_attention_1/dense_6/Tensordot/Prod:output:04window_attention_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
.window_attention_1/dense_6/Tensordot/transpose	Transpose%window_attention_1/Reshape_5:output:04window_attention_1/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
,window_attention_1/dense_6/Tensordot/ReshapeReshape2window_attention_1/dense_6/Tensordot/transpose:y:03window_attention_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
+window_attention_1/dense_6/Tensordot/MatMulMatMul5window_attention_1/dense_6/Tensordot/Reshape:output:0;window_attention_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
,window_attention_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2window_attention_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention_1/dense_6/Tensordot/concat_1ConcatV26window_attention_1/dense_6/Tensordot/GatherV2:output:05window_attention_1/dense_6/Tensordot/Const_2:output:0;window_attention_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
$window_attention_1/dense_6/TensordotReshape5window_attention_1/dense_6/Tensordot/MatMul:product:06window_attention_1/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"window_attention_1/dense_6/BiasAddBiasAdd-window_attention_1/dense_6/Tensordot:output:09window_attention_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@q
,window_attention_1/dropout_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
*window_attention_1/dropout_3/dropout_1/MulMul+window_attention_1/dense_6/BiasAdd:output:05window_attention_1/dropout_3/dropout_1/Const:output:0*
T0*+
_output_shapes
:?????????@?
,window_attention_1/dropout_3/dropout_1/ShapeShape+window_attention_1/dense_6/BiasAdd:output:0*
T0*
_output_shapes
:?
Cwindow_attention_1/dropout_3/dropout_1/random_uniform/RandomUniformRandomUniform5window_attention_1/dropout_3/dropout_1/Shape:output:0*
T0*+
_output_shapes
:?????????@*
dtype0z
5window_attention_1/dropout_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
3window_attention_1/dropout_3/dropout_1/GreaterEqualGreaterEqualLwindow_attention_1/dropout_3/dropout_1/random_uniform/RandomUniform:output:0>window_attention_1/dropout_3/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????@?
+window_attention_1/dropout_3/dropout_1/CastCast7window_attention_1/dropout_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????@?
,window_attention_1/dropout_3/dropout_1/Mul_1Mul.window_attention_1/dropout_3/dropout_1/Mul:z:0/window_attention_1/dropout_3/dropout_1/Cast:y:0*
T0*+
_output_shapes
:?????????@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   ?
	Reshape_4Reshape0window_attention_1/dropout_3/dropout_1/Mul_1:z:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:?????????@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!?????????@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:?????????  @]
Roll_1/shiftConst*
_output_shapes
:*
dtype0*
valueB"      \
Roll_1/axisConst*
_output_shapes
:*
dtype0*
valueB"      ?
Roll_1RollReshape_6:output:0Roll_1/shift:output:0Roll_1/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:?????????  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   v
	Reshape_7ReshapeRoll_1:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:??????????@S
drop_path_1/ShapeShapeReshape_7:output:0*
T0*
_output_shapes
:i
drop_path_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!drop_path_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!drop_path_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path_1/strided_sliceStridedSlicedrop_path_1/Shape:output:0(drop_path_1/strided_slice/stack:output:0*drop_path_1/strided_slice/stack_1:output:0*drop_path_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"drop_path_1/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d
"drop_path_1/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
 drop_path_1/random_uniform/shapePack"drop_path_1/strided_slice:output:0+drop_path_1/random_uniform/shape/1:output:0+drop_path_1/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:?
(drop_path_1/random_uniform/RandomUniformRandomUniform)drop_path_1/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0V
drop_path_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/addAddV2drop_path_1/add/x:output:01drop_path_1/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:?????????e
drop_path_1/FloorFloordrop_path_1/add:z:0*
T0*+
_output_shapes
:?????????Z
drop_path_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/truedivRealDivReshape_7:output:0drop_path_1/truediv/y:output:0*
T0*,
_output_shapes
:??????????@}
drop_path_1/mulMuldrop_path_1/truediv:z:0drop_path_1/Floor:y:0*
T0*,
_output_shapes
:??????????@[
addAddV2xdrop_path_1/mul:z:0*
T0*,
_output_shapes
:??????????@~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_3/moments/meanMeanadd:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/mul_1Muladd:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@?
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0m
#sequential_1/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential_1/dense_7/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential_1/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_7/Tensordot/GatherV2GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/free:output:05sequential_1/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/dense_7/Tensordot/GatherV2_1GatherV2-sequential_1/dense_7/Tensordot/Shape:output:0,sequential_1/dense_7/Tensordot/axes:output:07sequential_1/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????i
$sequential_1/activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"sequential_1/activation_1/Gelu/mulMul-sequential_1/activation_1/Gelu/mul/x:output:0%sequential_1/dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:???????????j
%sequential_1/activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
&sequential_1/activation_1/Gelu/truedivRealDiv%sequential_1/dense_7/BiasAdd:output:0.sequential_1/activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:????????????
"sequential_1/activation_1/Gelu/ErfErf*sequential_1/activation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????i
$sequential_1/activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"sequential_1/activation_1/Gelu/addAddV2-sequential_1/activation_1/Gelu/add/x:output:0&sequential_1/activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
$sequential_1/activation_1/Gelu/mul_1Mul&sequential_1/activation_1/Gelu/mul:z:0&sequential_1/activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:???????????i
$sequential_1/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
"sequential_1/dropout_4/dropout/MulMul(sequential_1/activation_1/Gelu/mul_1:z:0-sequential_1/dropout_4/dropout/Const:output:0*
T0*-
_output_shapes
:???????????|
$sequential_1/dropout_4/dropout/ShapeShape(sequential_1/activation_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:?
;sequential_1/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_4/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0r
-sequential_1/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
+sequential_1/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_4/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
#sequential_1/dropout_4/dropout/CastCast/sequential_1/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
$sequential_1/dropout_4/dropout/Mul_1Mul&sequential_1/dropout_4/dropout/Mul:z:0'sequential_1/dropout_4/dropout/Cast:y:0*
T0*-
_output_shapes
:????????????
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0m
#sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
$sequential_1/dense_8/Tensordot/ShapeShape(sequential_1/dropout_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:n
,sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_8/Tensordot/GatherV2GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/free:output:05sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_1/dense_8/Tensordot/GatherV2_1GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/axes:output:07sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
(sequential_1/dense_8/Tensordot/transpose	Transpose(sequential_1/dropout_4/dropout/Mul_1:z:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@i
$sequential_1/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
"sequential_1/dropout_5/dropout/MulMul%sequential_1/dense_8/BiasAdd:output:0-sequential_1/dropout_5/dropout/Const:output:0*
T0*,
_output_shapes
:??????????@y
$sequential_1/dropout_5/dropout/ShapeShape%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:?
;sequential_1/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_5/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype0r
-sequential_1/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
+sequential_1/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_5/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@?
#sequential_1/dropout_5/dropout/CastCast/sequential_1/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@?
$sequential_1/dropout_5/dropout/Mul_1Mul&sequential_1/dropout_5/dropout/Mul:z:0'sequential_1/dropout_5/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@k
drop_path_1/Shape_1Shape(sequential_1/dropout_5/dropout/Mul_1:z:0*
T0*
_output_shapes
:k
!drop_path_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#drop_path_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#drop_path_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path_1/strided_slice_1StridedSlicedrop_path_1/Shape_1:output:0*drop_path_1/strided_slice_1/stack:output:0,drop_path_1/strided_slice_1/stack_1:output:0,drop_path_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$drop_path_1/random_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :f
$drop_path_1/random_uniform_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
"drop_path_1/random_uniform_1/shapePack$drop_path_1/strided_slice_1:output:0-drop_path_1/random_uniform_1/shape/1:output:0-drop_path_1/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
*drop_path_1/random_uniform_1/RandomUniformRandomUniform+drop_path_1/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0X
drop_path_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/add_1AddV2drop_path_1/add_1/x:output:03drop_path_1/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:?????????i
drop_path_1/Floor_1Floordrop_path_1/add_1:z:0*
T0*+
_output_shapes
:?????????\
drop_path_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path_1/truediv_1RealDiv(sequential_1/dropout_5/dropout/Mul_1:z:0 drop_path_1/truediv_1/y:output:0*
T0*,
_output_shapes
:??????????@?
drop_path_1/mul_1Muldrop_path_1/truediv_1:z:0drop_path_1/Floor_1:y:0*
T0*,
_output_shapes
:??????????@e
add_1AddV2add:z:0drop_path_1/mul_1:z:0*
T0*,
_output_shapes
:??????????@?
NoOpNoOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp/^window_attention_1/ExpandDims_1/ReadVariableOp^window_attention_1/Gather,^window_attention_1/Reshape_1/ReadVariableOp2^window_attention_1/dense_5/BiasAdd/ReadVariableOp4^window_attention_1/dense_5/Tensordot/ReadVariableOp2^window_attention_1/dense_6/BiasAdd/ReadVariableOp4^window_attention_1/dense_6/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????@: : : : : : : : : : : : : : : 2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2^
-sequential_1/dense_7/Tensordot/ReadVariableOp-sequential_1/dense_7/Tensordot/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2^
-sequential_1/dense_8/Tensordot/ReadVariableOp-sequential_1/dense_8/Tensordot/ReadVariableOp2`
.window_attention_1/ExpandDims_1/ReadVariableOp.window_attention_1/ExpandDims_1/ReadVariableOp26
window_attention_1/Gatherwindow_attention_1/Gather2Z
+window_attention_1/Reshape_1/ReadVariableOp+window_attention_1/Reshape_1/ReadVariableOp2f
1window_attention_1/dense_5/BiasAdd/ReadVariableOp1window_attention_1/dense_5/BiasAdd/ReadVariableOp2j
3window_attention_1/dense_5/Tensordot/ReadVariableOp3window_attention_1/dense_5/Tensordot/ReadVariableOp2f
1window_attention_1/dense_6/BiasAdd/ReadVariableOp1window_attention_1/dense_6/BiasAdd/ReadVariableOp2j
3window_attention_1/dense_6/Tensordot/ReadVariableOp3window_attention_1/dense_6/Tensordot/ReadVariableOp:O K
,
_output_shapes
:??????????@

_user_specified_namex
?G
?
E__inference_sequential_layer_call_and_return_conditional_losses_12820

inputs<
)dense_3_tensordot_readvariableop_resource:	@?6
'dense_3_biasadd_readvariableop_resource:	?<
)dense_4_tensordot_readvariableop_resource:	?@5
'dense_4_biasadd_readvariableop_resource:@
identity??dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/Tensordot/ReadVariableOp?
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       M
dense_3/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_3/Tensordot/transpose	Transposeinputs!dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????Z
activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
activation/Gelu/mulMulactivation/Gelu/mul/x:output:0dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:???????????[
activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
activation/Gelu/truedivRealDivdense_3/BiasAdd:output:0activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:???????????o
activation/Gelu/ErfErfactivation/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????Z
activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
activation/Gelu/addAddV2activation/Gelu/add/x:output:0activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
activation/Gelu/mul_1Mulactivation/Gelu/mul:z:0activation/Gelu/add:z:0*
T0*-
_output_shapes
:???????????q
dropout_1/IdentityIdentityactivation/Gelu/mul_1:z:0*
T0*-
_output_shapes
:????????????
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_4/Tensordot/ShapeShapedropout_1/Identity:output:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_4/Tensordot/transpose	Transposedropout_1/Identity:output:0!dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@o
dropout_2/IdentityIdentitydense_4/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@o
IdentityIdentitydropout_2/Identity:output:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_12406
dense_3_input 
dense_3_12392:	@?
dense_3_12394:	? 
dense_4_12399:	?@
dense_4_12401:@
identity??dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCalldense_3_inputdense_3_12392dense_3_12394*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_12162?
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12180?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12297?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_4_12399dense_4_12401*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_12219?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12264~
IdentityIdentity*dropout_2/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:[ W
,
_output_shapes
:??????????@
'
_user_specified_namedense_3_input
?
?
1__inference_swin_transformer_1_layer_call_fn_1623
x
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	
	unknown_4:	 
	unknown_5:?
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:	@?

unknown_11:	?

unknown_12:	?@

unknown_13:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_1603`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????@: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
?
.__inference_patch_embedding_layer_call_fn_1272	
patch
unknown:@
	unknown_0:@
	unknown_1:	?@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallpatchunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_patch_embedding_layer_call_and_return_conditional_losses_1264`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:??????????

_user_specified_namepatch
?
H
,__inference_patch_extract_layer_call_fn_3752

images
identity?
PartitionedCallPartitionedCallimages*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_patch_extract_layer_call_and_return_conditional_losses_3697e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameimages
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_12389
dense_3_input 
dense_3_12375:	@?
dense_3_12377:	? 
dense_4_12382:	?@
dense_4_12384:@
identity??dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCalldense_3_inputdense_3_12375dense_3_12377*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_12162?
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12180?
dropout_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12187?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_4_12382dense_4_12384*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_12219?
dropout_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12230v
IdentityIdentity"dropout_2/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:[ W
,
_output_shapes
:??????????@
'
_user_specified_namedense_3_input
?
b
)__inference_dropout_2_layer_call_fn_13204

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12264t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
(__inference_restored_function_body_10686
x
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	
	unknown_4:	 
	unknown_5:?
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:	@?

unknown_11:	?

unknown_12:	?@

unknown_13:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_6078t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????@: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_13209

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
x
'__inference_restored_function_body_9927
x
unknown:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown*
Tin
2*
Tout
2*-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_patch_merging_layer_call_and_return_conditional_losses_2938u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????@: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
?
%__inference_model_layer_call_fn_10178
input_1
unknown:@
	unknown_0:@
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:	@?
	unknown_5:	?
	unknown_6:	
	unknown_7:	
	unknown_8:@@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:	@?

unknown_13:	?

unknown_14:	?@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@?

unknown_19:	?

unknown_20:	

unknown_21:	!

unknown_22:?

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:	@?

unknown_28:	?

unknown_29:	?@

unknown_30:@

unknown_31:
??

unknown_32:	?W

unknown_33:W
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*E
_read_only_resource_inputs'
%#	
 !"#*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
?
,__inference_sequential_1_layer_call_fn_12913

inputs
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_12514t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
(__inference_restored_function_body_10622
x
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	
	unknown_4:	
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:	@?

unknown_10:	?

unknown_11:	?@

unknown_12:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2087t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12670
dense_7_input 
dense_7_12656:	@?
dense_7_12658:	? 
dense_8_12663:	?@
dense_8_12665:@
identity??dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCalldense_7_inputdense_7_12656dense_7_12658*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_12443?
activation_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_12461?
dropout_4/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12468?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_8_12663dense_8_12665*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_12500?
dropout_5/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12511v
IdentityIdentity"dropout_5/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:[ W
,
_output_shapes
:??????????@
'
_user_specified_namedense_7_input
?
?
%__inference_model_layer_call_fn_11285

inputs
unknown:	
	unknown_0:	
	unknown_1:@
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@
	unknown_6:	@?
	unknown_7:	?
	unknown_8:	
	unknown_9:	

unknown_10:@@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:	@?

unknown_15:	?

unknown_16:	?@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:	@?

unknown_21:	?

unknown_22:	

unknown_23:	!

unknown_24:?

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:	@?

unknown_30:	?

unknown_31:	?@

unknown_32:@

unknown_33:
??

unknown_34:	?W

unknown_35:W
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*E
_read_only_resource_inputs'
%#	
 !"#$%*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10728o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
J__inference_swin_transformer_layer_call_and_return_conditional_losses_6423
xG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@M
:window_attention_dense_1_tensordot_readvariableop_resource:	@?G
8window_attention_dense_1_biasadd_readvariableop_resource:	?D
2window_attention_reshape_1_readvariableop_resource:	2
 window_attention_gather_resource:	L
:window_attention_dense_2_tensordot_readvariableop_resource:@@F
8window_attention_dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@G
4sequential_dense_3_tensordot_readvariableop_resource:	@?A
2sequential_dense_3_biasadd_readvariableop_resource:	?G
4sequential_dense_4_tensordot_readvariableop_resource:	?@@
2sequential_dense_4_biasadd_readvariableop_resource:@
identity??,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?)sequential/dense_3/BiasAdd/ReadVariableOp?+sequential/dense_3/Tensordot/ReadVariableOp?)sequential/dense_4/BiasAdd/ReadVariableOp?+sequential/dense_4/Tensordot/ReadVariableOp?window_attention/Gather?)window_attention/Reshape_1/ReadVariableOp?/window_attention/dense_1/BiasAdd/ReadVariableOp?1window_attention/dense_1/Tensordot/ReadVariableOp?/window_attention/dense_2/BiasAdd/ReadVariableOp?1window_attention/dense_2/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanx;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:???????????
-layer_normalization/moments/SquaredDifferenceSquaredDifferencex1layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/mul_1Mulx%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   ?
ReshapeReshape'layer_normalization/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:?????????  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_1ReshapeReshape:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!?????????@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention/dense_1/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0q
'window_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'window_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       j
(window_attention/dense_1/Tensordot/ShapeShapeReshape_3:output:0*
T0*
_output_shapes
:r
0window_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_1/Tensordot/GatherV2GatherV21window_attention/dense_1/Tensordot/Shape:output:00window_attention/dense_1/Tensordot/free:output:09window_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2window_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention/dense_1/Tensordot/GatherV2_1GatherV21window_attention/dense_1/Tensordot/Shape:output:00window_attention/dense_1/Tensordot/axes:output:0;window_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(window_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
'window_attention/dense_1/Tensordot/ProdProd4window_attention/dense_1/Tensordot/GatherV2:output:01window_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention/dense_1/Tensordot/Prod_1Prod6window_attention/dense_1/Tensordot/GatherV2_1:output:03window_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)window_attention/dense_1/Tensordot/concatConcatV20window_attention/dense_1/Tensordot/free:output:00window_attention/dense_1/Tensordot/axes:output:07window_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(window_attention/dense_1/Tensordot/stackPack0window_attention/dense_1/Tensordot/Prod:output:02window_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
,window_attention/dense_1/Tensordot/transpose	TransposeReshape_3:output:02window_attention/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
*window_attention/dense_1/Tensordot/ReshapeReshape0window_attention/dense_1/Tensordot/transpose:y:01window_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
)window_attention/dense_1/Tensordot/MatMulMatMul3window_attention/dense_1/Tensordot/Reshape:output:09window_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????u
*window_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?r
0window_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_1/Tensordot/concat_1ConcatV24window_attention/dense_1/Tensordot/GatherV2:output:03window_attention/dense_1/Tensordot/Const_2:output:09window_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
"window_attention/dense_1/TensordotReshape3window_attention/dense_1/Tensordot/MatMul:product:04window_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:???????????
/window_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
 window_attention/dense_1/BiasAddBiasAdd+window_attention/dense_1/Tensordot:output:07window_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????{
window_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention/ReshapeReshape)window_attention/dense_1/BiasAdd:output:0'window_attention/Reshape/shape:output:0*
T0*3
_output_shapes!
:?????????|
window_attention/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
window_attention/transpose	Transpose!window_attention/Reshape:output:0(window_attention/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????n
$window_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&window_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&window_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
window_attention/strided_sliceStridedSlicewindow_attention/transpose:y:0-window_attention/strided_slice/stack:output:0/window_attention/strided_slice/stack_1:output:0/window_attention/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskp
&window_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention/strided_slice_1StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_1/stack:output:01window_attention/strided_slice_1/stack_1:output:01window_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskp
&window_attention/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention/strided_slice_2StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_2/stack:output:01window_attention/strided_slice_2/stack_1:output:01window_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_mask[
window_attention/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
window_attention/mulMul'window_attention/strided_slice:output:0window_attention/mul/y:output:0*
T0*/
_output_shapes
:?????????z
!window_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention/transpose_1	Transpose)window_attention/strided_slice_1:output:0*window_attention/transpose_1/perm:output:0*
T0*/
_output_shapes
:??????????
window_attention/matmulBatchMatMulV2window_attention/mul:z:0 window_attention/transpose_1:y:0*
T0*/
_output_shapes
:??????????
)window_attention/Reshape_1/ReadVariableOpReadVariableOp2window_attention_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	s
 window_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
window_attention/Reshape_1Reshape1window_attention/Reshape_1/ReadVariableOp:value:0)window_attention/Reshape_1/shape:output:0*
T0	*
_output_shapes
:?
window_attention/GatherResourceGather window_attention_gather_resource#window_attention/Reshape_1:output:0*
Tindices0	*
_output_shapes

:*
dtype0p
window_attention/IdentityIdentity window_attention/Gather:output:0*
T0*
_output_shapes

:u
 window_attention/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?????
window_attention/Reshape_2Reshape"window_attention/Identity:output:0)window_attention/Reshape_2/shape:output:0*
T0*"
_output_shapes
:v
!window_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
window_attention/transpose_2	Transpose#window_attention/Reshape_2:output:0*window_attention/transpose_2/perm:output:0*
T0*"
_output_shapes
:a
window_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention/ExpandDims
ExpandDims window_attention/transpose_2:y:0(window_attention/ExpandDims/dim:output:0*
T0*&
_output_shapes
:?
window_attention/addAddV2 window_attention/matmul:output:0$window_attention/ExpandDims:output:0*
T0*/
_output_shapes
:?????????w
window_attention/SoftmaxSoftmaxwindow_attention/add:z:0*
T0*/
_output_shapes
:??????????
!window_attention/dropout/IdentityIdentity"window_attention/Softmax:softmax:0*
T0*/
_output_shapes
:??????????
window_attention/matmul_1BatchMatMulV2*window_attention/dropout/Identity:output:0)window_attention/strided_slice_2:output:0*
T0*/
_output_shapes
:?????????z
!window_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention/transpose_3	Transpose"window_attention/matmul_1:output:0*window_attention/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????u
 window_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   ?
window_attention/Reshape_3Reshape window_attention/transpose_3:y:0)window_attention/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention/dense_2/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0q
'window_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'window_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
(window_attention/dense_2/Tensordot/ShapeShape#window_attention/Reshape_3:output:0*
T0*
_output_shapes
:r
0window_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_2/Tensordot/GatherV2GatherV21window_attention/dense_2/Tensordot/Shape:output:00window_attention/dense_2/Tensordot/free:output:09window_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2window_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention/dense_2/Tensordot/GatherV2_1GatherV21window_attention/dense_2/Tensordot/Shape:output:00window_attention/dense_2/Tensordot/axes:output:0;window_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(window_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
'window_attention/dense_2/Tensordot/ProdProd4window_attention/dense_2/Tensordot/GatherV2:output:01window_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention/dense_2/Tensordot/Prod_1Prod6window_attention/dense_2/Tensordot/GatherV2_1:output:03window_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)window_attention/dense_2/Tensordot/concatConcatV20window_attention/dense_2/Tensordot/free:output:00window_attention/dense_2/Tensordot/axes:output:07window_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(window_attention/dense_2/Tensordot/stackPack0window_attention/dense_2/Tensordot/Prod:output:02window_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
,window_attention/dense_2/Tensordot/transpose	Transpose#window_attention/Reshape_3:output:02window_attention/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
*window_attention/dense_2/Tensordot/ReshapeReshape0window_attention/dense_2/Tensordot/transpose:y:01window_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
)window_attention/dense_2/Tensordot/MatMulMatMul3window_attention/dense_2/Tensordot/Reshape:output:09window_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@t
*window_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@r
0window_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_2/Tensordot/concat_1ConcatV24window_attention/dense_2/Tensordot/GatherV2:output:03window_attention/dense_2/Tensordot/Const_2:output:09window_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
"window_attention/dense_2/TensordotReshape3window_attention/dense_2/Tensordot/MatMul:product:04window_attention/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@?
/window_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
 window_attention/dense_2/BiasAddBiasAdd+window_attention/dense_2/Tensordot:output:07window_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@?
#window_attention/dropout/Identity_1Identity)window_attention/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   ?
	Reshape_4Reshape,window_attention/dropout/Identity_1:output:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:?????????@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!?????????@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:?????????  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   y
	Reshape_7ReshapeReshape_6:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:??????????@Q
drop_path/ShapeShapeReshape_7:output:0*
T0*
_output_shapes
:g
drop_path/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
drop_path/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
drop_path/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path/strided_sliceStridedSlicedrop_path/Shape:output:0&drop_path/strided_slice/stack:output:0(drop_path/strided_slice/stack_1:output:0(drop_path/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 drop_path/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :b
 drop_path/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
drop_path/random_uniform/shapePack drop_path/strided_slice:output:0)drop_path/random_uniform/shape/1:output:0)drop_path/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:?
&drop_path/random_uniform/RandomUniformRandomUniform'drop_path/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0T
drop_path/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/addAddV2drop_path/add/x:output:0/drop_path/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:?????????a
drop_path/FloorFloordrop_path/add:z:0*
T0*+
_output_shapes
:?????????X
drop_path/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/truedivRealDivReshape_7:output:0drop_path/truediv/y:output:0*
T0*,
_output_shapes
:??????????@w
drop_path/mulMuldrop_path/truediv:z:0drop_path/Floor:y:0*
T0*,
_output_shapes
:??????????@Y
addAddV2xdrop_path/mul:z:0*
T0*,
_output_shapes
:??????????@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@?
+sequential/dense_3/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0k
!sequential/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
"sequential/dense_3/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_3/Tensordot/GatherV2GatherV2+sequential/dense_3/Tensordot/Shape:output:0*sequential/dense_3/Tensordot/free:output:03sequential/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential/dense_3/Tensordot/GatherV2_1GatherV2+sequential/dense_3/Tensordot/Shape:output:0*sequential/dense_3/Tensordot/axes:output:05sequential/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!sequential/dense_3/Tensordot/ProdProd.sequential/dense_3/Tensordot/GatherV2:output:0+sequential/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
#sequential/dense_3/Tensordot/Prod_1Prod0sequential/dense_3/Tensordot/GatherV2_1:output:0-sequential/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/dense_3/Tensordot/concatConcatV2*sequential/dense_3/Tensordot/free:output:0*sequential/dense_3/Tensordot/axes:output:01sequential/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"sequential/dense_3/Tensordot/stackPack*sequential/dense_3/Tensordot/Prod:output:0,sequential/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
&sequential/dense_3/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0,sequential/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
$sequential/dense_3/Tensordot/ReshapeReshape*sequential/dense_3/Tensordot/transpose:y:0+sequential/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
#sequential/dense_3/Tensordot/MatMulMatMul-sequential/dense_3/Tensordot/Reshape:output:03sequential/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
$sequential/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?l
*sequential/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_3/Tensordot/concat_1ConcatV2.sequential/dense_3/Tensordot/GatherV2:output:0-sequential/dense_3/Tensordot/Const_2:output:03sequential/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential/dense_3/TensordotReshape-sequential/dense_3/Tensordot/MatMul:product:0.sequential/dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/dense_3/BiasAddBiasAdd%sequential/dense_3/Tensordot:output:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????e
 sequential/activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
sequential/activation/Gelu/mulMul)sequential/activation/Gelu/mul/x:output:0#sequential/dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:???????????f
!sequential/activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
"sequential/activation/Gelu/truedivRealDiv#sequential/dense_3/BiasAdd:output:0*sequential/activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:????????????
sequential/activation/Gelu/ErfErf&sequential/activation/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????e
 sequential/activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sequential/activation/Gelu/addAddV2)sequential/activation/Gelu/add/x:output:0"sequential/activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
 sequential/activation/Gelu/mul_1Mul"sequential/activation/Gelu/mul:z:0"sequential/activation/Gelu/add:z:0*
T0*-
_output_shapes
:????????????
sequential/dropout_1/IdentityIdentity$sequential/activation/Gelu/mul_1:z:0*
T0*-
_output_shapes
:????????????
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0k
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       x
"sequential/dense_4/Tensordot/ShapeShape&sequential/dropout_1/Identity:output:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
&sequential/dense_4/Tensordot/transpose	Transpose&sequential/dropout_1/Identity:output:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
sequential/dropout_2/IdentityIdentity#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@g
drop_path/Shape_1Shape&sequential/dropout_2/Identity:output:0*
T0*
_output_shapes
:i
drop_path/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!drop_path/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!drop_path/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path/strided_slice_1StridedSlicedrop_path/Shape_1:output:0(drop_path/strided_slice_1/stack:output:0*drop_path/strided_slice_1/stack_1:output:0*drop_path/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"drop_path/random_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d
"drop_path/random_uniform_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
 drop_path/random_uniform_1/shapePack"drop_path/strided_slice_1:output:0+drop_path/random_uniform_1/shape/1:output:0+drop_path/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
(drop_path/random_uniform_1/RandomUniformRandomUniform)drop_path/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0V
drop_path/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/add_1AddV2drop_path/add_1/x:output:01drop_path/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:?????????e
drop_path/Floor_1Floordrop_path/add_1:z:0*
T0*+
_output_shapes
:?????????Z
drop_path/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/truediv_1RealDiv&sequential/dropout_2/Identity:output:0drop_path/truediv_1/y:output:0*
T0*,
_output_shapes
:??????????@}
drop_path/mul_1Muldrop_path/truediv_1:z:0drop_path/Floor_1:y:0*
T0*,
_output_shapes
:??????????@c
add_1AddV2add:z:0drop_path/mul_1:z:0*
T0*,
_output_shapes
:??????????@?
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp,^sequential/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp^window_attention/Gather*^window_attention/Reshape_1/ReadVariableOp0^window_attention/dense_1/BiasAdd/ReadVariableOp2^window_attention/dense_1/Tensordot/ReadVariableOp0^window_attention/dense_2/BiasAdd/ReadVariableOp2^window_attention/dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????@: : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2Z
+sequential/dense_3/Tensordot/ReadVariableOp+sequential/dense_3/Tensordot/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp22
window_attention/Gatherwindow_attention/Gather2V
)window_attention/Reshape_1/ReadVariableOp)window_attention/Reshape_1/ReadVariableOp2b
/window_attention/dense_1/BiasAdd/ReadVariableOp/window_attention/dense_1/BiasAdd/ReadVariableOp2f
1window_attention/dense_1/Tensordot/ReadVariableOp1window_attention/dense_1/Tensordot/ReadVariableOp2b
/window_attention/dense_2/BiasAdd/ReadVariableOp/window_attention/dense_2/BiasAdd/ReadVariableOp2f
1window_attention/dense_2/Tensordot/ReadVariableOp1window_attention/dense_2/Tensordot/ReadVariableOp:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_13292

inputs

identity_1T
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????a

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
C__inference_dense_10_layer_call_and_return_conditional_losses_12718

inputs1
matmul_readvariableop_resource:	?W-
biasadd_readvariableop_resource:W
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?W*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Wr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:W*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????WV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????W`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????Ww
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?d
?
__inference__wrapped_model_9941
input_1,
model_patch_embedding_9786:@(
model_patch_embedding_9788:@-
model_patch_embedding_9790:	?@)
model_swin_transformer_9826:@)
model_swin_transformer_9828:@.
model_swin_transformer_9830:	@?*
model_swin_transformer_9832:	?-
model_swin_transformer_9834:	-
model_swin_transformer_9836:	-
model_swin_transformer_9838:@@)
model_swin_transformer_9840:@)
model_swin_transformer_9842:@)
model_swin_transformer_9844:@.
model_swin_transformer_9846:	@?*
model_swin_transformer_9848:	?.
model_swin_transformer_9850:	?@)
model_swin_transformer_9852:@+
model_swin_transformer_1_9890:@+
model_swin_transformer_1_9892:@0
model_swin_transformer_1_9894:	@?,
model_swin_transformer_1_9896:	?/
model_swin_transformer_1_9898:	/
model_swin_transformer_1_9900:	4
model_swin_transformer_1_9902:?/
model_swin_transformer_1_9904:@@+
model_swin_transformer_1_9906:@+
model_swin_transformer_1_9908:@+
model_swin_transformer_1_9910:@0
model_swin_transformer_1_9912:	@?,
model_swin_transformer_1_9914:	?0
model_swin_transformer_1_9916:	?@+
model_swin_transformer_1_9918:@,
model_patch_merging_9928:
??@
-model_dense_10_matmul_readvariableop_resource:	?W<
.model_dense_10_biasadd_readvariableop_resource:W
identity??%model/dense_10/BiasAdd/ReadVariableOp?$model/dense_10/MatMul/ReadVariableOp?-model/patch_embedding/StatefulPartitionedCall?+model/patch_merging/StatefulPartitionedCall?.model/swin_transformer/StatefulPartitionedCall?0model/swin_transformer_1/StatefulPartitionedCallN
model/random_crop/ShapeShapeinput_1*
T0*
_output_shapes
:x
%model/random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????z
'model/random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????q
'model/random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model/random_crop/strided_sliceStridedSlice model/random_crop/Shape:output:0.model/random_crop/strided_slice/stack:output:00model/random_crop/strided_slice/stack_1:output:00model/random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
'model/random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
)model/random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????s
)model/random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!model/random_crop/strided_slice_1StridedSlice model/random_crop/Shape:output:00model/random_crop/strided_slice_1/stack:output:02model/random_crop/strided_slice_1/stack_1:output:02model/random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
model/random_crop/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@?
model/random_crop/mulMul*model/random_crop/strided_slice_1:output:0 model/random_crop/mul/y:output:0*
T0*
_output_shapes
: i
model/random_crop/CastCastmodel/random_crop/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: `
model/random_crop/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B?
model/random_crop/truedivRealDivmodel/random_crop/Cast:y:0$model/random_crop/truediv/y:output:0*
T0*
_output_shapes
: o
model/random_crop/Cast_1Castmodel/random_crop/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: [
model/random_crop/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :@?
model/random_crop/mul_1Mul(model/random_crop/strided_slice:output:0"model/random_crop/mul_1/y:output:0*
T0*
_output_shapes
: m
model/random_crop/Cast_2Castmodel/random_crop/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: b
model/random_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B?
model/random_crop/truediv_1RealDivmodel/random_crop/Cast_2:y:0&model/random_crop/truediv_1/y:output:0*
T0*
_output_shapes
: q
model/random_crop/Cast_3Castmodel/random_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
model/random_crop/MinimumMinimum(model/random_crop/strided_slice:output:0model/random_crop/Cast_1:y:0*
T0*
_output_shapes
: ?
model/random_crop/Minimum_1Minimum*model/random_crop/strided_slice_1:output:0model/random_crop/Cast_3:y:0*
T0*
_output_shapes
: ?
model/random_crop/subSub(model/random_crop/strided_slice:output:0model/random_crop/Minimum:z:0*
T0*
_output_shapes
: k
model/random_crop/Cast_4Castmodel/random_crop/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: b
model/random_crop/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
model/random_crop/truediv_2RealDivmodel/random_crop/Cast_4:y:0&model/random_crop/truediv_2/y:output:0*
T0*
_output_shapes
: q
model/random_crop/Cast_5Castmodel/random_crop/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
model/random_crop/sub_1Sub*model/random_crop/strided_slice_1:output:0model/random_crop/Minimum_1:z:0*
T0*
_output_shapes
: m
model/random_crop/Cast_6Castmodel/random_crop/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: b
model/random_crop/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
model/random_crop/truediv_3RealDivmodel/random_crop/Cast_6:y:0&model/random_crop/truediv_3/y:output:0*
T0*
_output_shapes
: q
model/random_crop/Cast_7Castmodel/random_crop/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: [
model/random_crop/stack/0Const*
_output_shapes
: *
dtype0*
value	B : [
model/random_crop/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
model/random_crop/stackPack"model/random_crop/stack/0:output:0model/random_crop/Cast_5:y:0model/random_crop/Cast_7:y:0"model/random_crop/stack/3:output:0*
N*
T0*
_output_shapes
:f
model/random_crop/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????f
model/random_crop/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
model/random_crop/stack_1Pack$model/random_crop/stack_1/0:output:0model/random_crop/Minimum:z:0model/random_crop/Minimum_1:z:0$model/random_crop/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
model/random_crop/SliceSliceinput_1 model/random_crop/stack:output:0"model/random_crop/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????@@?????????n
model/random_crop/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   ?
'model/random_crop/resize/ResizeBilinearResizeBilinear model/random_crop/Slice:output:0&model/random_crop/resize/size:output:0*
T0*/
_output_shapes
:?????????@@*
half_pixel_centers(?
#model/patch_extract/PartitionedCallPartitionedCall8model/random_crop/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9773?
-model/patch_embedding/StatefulPartitionedCallStatefulPartitionedCall,model/patch_extract/PartitionedCall:output:0model_patch_embedding_9786model_patch_embedding_9788model_patch_embedding_9790*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9785?
.model/swin_transformer/StatefulPartitionedCallStatefulPartitionedCall6model/patch_embedding/StatefulPartitionedCall:output:0model_swin_transformer_9826model_swin_transformer_9828model_swin_transformer_9830model_swin_transformer_9832model_swin_transformer_9834model_swin_transformer_9836model_swin_transformer_9838model_swin_transformer_9840model_swin_transformer_9842model_swin_transformer_9844model_swin_transformer_9846model_swin_transformer_9848model_swin_transformer_9850model_swin_transformer_9852*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9825?
0model/swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall7model/swin_transformer/StatefulPartitionedCall:output:0model_swin_transformer_1_9890model_swin_transformer_1_9892model_swin_transformer_1_9894model_swin_transformer_1_9896model_swin_transformer_1_9898model_swin_transformer_1_9900model_swin_transformer_1_9902model_swin_transformer_1_9904model_swin_transformer_1_9906model_swin_transformer_1_9908model_swin_transformer_1_9910model_swin_transformer_1_9912model_swin_transformer_1_9914model_swin_transformer_1_9916model_swin_transformer_1_9918*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9889?
+model/patch_merging/StatefulPartitionedCallStatefulPartitionedCall9model/swin_transformer_1/StatefulPartitionedCall:output:0model_patch_merging_9928*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9927w
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
#model/global_average_pooling1d/MeanMean4model/patch_merging/StatefulPartitionedCall:output:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
$model/dense_10/MatMul/ReadVariableOpReadVariableOp-model_dense_10_matmul_readvariableop_resource*
_output_shapes
:	?W*
dtype0?
model/dense_10/MatMulMatMul,model/global_average_pooling1d/Mean:output:0,model/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????W?
%model/dense_10/BiasAdd/ReadVariableOpReadVariableOp.model_dense_10_biasadd_readvariableop_resource*
_output_shapes
:W*
dtype0?
model/dense_10/BiasAddBiasAddmodel/dense_10/MatMul:product:0-model/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Wt
model/dense_10/SoftmaxSoftmaxmodel/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Wo
IdentityIdentity model/dense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????W?
NoOpNoOp&^model/dense_10/BiasAdd/ReadVariableOp%^model/dense_10/MatMul/ReadVariableOp.^model/patch_embedding/StatefulPartitionedCall,^model/patch_merging/StatefulPartitionedCall/^model/swin_transformer/StatefulPartitionedCall1^model/swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%model/dense_10/BiasAdd/ReadVariableOp%model/dense_10/BiasAdd/ReadVariableOp2L
$model/dense_10/MatMul/ReadVariableOp$model/dense_10/MatMul/ReadVariableOp2^
-model/patch_embedding/StatefulPartitionedCall-model/patch_embedding/StatefulPartitionedCall2Z
+model/patch_merging/StatefulPartitionedCall+model/patch_merging/StatefulPartitionedCall2`
.model/swin_transformer/StatefulPartitionedCall.model/swin_transformer/StatefulPartitionedCall2d
0model/swin_transformer_1/StatefulPartitionedCall0model/swin_transformer_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?"
b
F__inference_random_crop_layer_call_and_return_conditional_losses_11827

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :@U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: E
CastCastmul:z:0*

DstT0*

SrcT0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?BQ
truedivRealDivCast:y:0truediv/y:output:0*
T0*
_output_shapes
: K
Cast_1Casttruediv:z:0*

DstT0*

SrcT0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :@W
mul_1Mulstrided_slice:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
Cast_2Cast	mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?BW
	truediv_1RealDiv
Cast_2:y:0truediv_1/y:output:0*
T0*
_output_shapes
: M
Cast_3Casttruediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: W
MinimumMinimumstrided_slice:output:0
Cast_1:y:0*
T0*
_output_shapes
: [
	Minimum_1Minimumstrided_slice_1:output:0
Cast_3:y:0*
T0*
_output_shapes
: P
subSubstrided_slice:output:0Minimum:z:0*
T0*
_output_shapes
: G
Cast_4Castsub:z:0*

DstT0*

SrcT0*
_output_shapes
: P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @W
	truediv_2RealDiv
Cast_4:y:0truediv_2/y:output:0*
T0*
_output_shapes
: M
Cast_5Casttruediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: V
sub_1Substrided_slice_1:output:0Minimum_1:z:0*
T0*
_output_shapes
: I
Cast_6Cast	sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @W
	truediv_3RealDiv
Cast_6:y:0truediv_3/y:output:0*
T0*
_output_shapes
: M
Cast_7Casttruediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: I
stack/0Const*
_output_shapes
: *
dtype0*
value	B : I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : w
stackPackstack/0:output:0
Cast_5:y:0
Cast_7:y:0stack/3:output:0*
N*
T0*
_output_shapes
:T
	stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????T
	stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
stack_1Packstack_1/0:output:0Minimum:z:0Minimum_1:z:0stack_1/3:output:0*
N*
T0*
_output_shapes
:?
SliceSliceinputsstack:output:0stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????@@?????????\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   ?
resize/ResizeBilinearResizeBilinearSlice:output:0resize/size:output:0*
T0*/
_output_shapes
:?????????@@*
half_pixel_centers(v
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
B__inference_dense_4_layer_call_and_return_conditional_losses_13194

inputs4
!tensordot_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:????????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?X
?
__inference__traced_save_13522
file_prefix:
6savev2_swin_transformer_1_variable_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop;
7savev2_patch_embedding_dense_kernel_read_readvariableop9
5savev2_patch_embedding_dense_bias_read_readvariableopC
?savev2_patch_embedding_embedding_embeddings_read_readvariableopI
Esavev2_swin_transformer_layer_normalization_gamma_read_readvariableopH
Dsavev2_swin_transformer_layer_normalization_beta_read_readvariableopG
Csavev2_swin_transformer_window_attention_weight_read_readvariableopO
Ksavev2_swin_transformer_window_attention_dense_1_kernel_read_readvariableopM
Isavev2_swin_transformer_window_attention_dense_1_bias_read_readvariableopO
Ksavev2_swin_transformer_window_attention_dense_2_kernel_read_readvariableopM
Isavev2_swin_transformer_window_attention_dense_2_bias_read_readvariableopK
Gsavev2_swin_transformer_layer_normalization_1_gamma_read_readvariableopJ
Fsavev2_swin_transformer_layer_normalization_1_beta_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableopI
Esavev2_swin_transformer_window_attention_variable_read_readvariableop	M
Isavev2_swin_transformer_1_layer_normalization_2_gamma_read_readvariableopL
Hsavev2_swin_transformer_1_layer_normalization_2_beta_read_readvariableopK
Gsavev2_swin_transformer_1_window_attention_1_weight_read_readvariableopS
Osavev2_swin_transformer_1_window_attention_1_dense_5_kernel_read_readvariableopQ
Msavev2_swin_transformer_1_window_attention_1_dense_5_bias_read_readvariableopS
Osavev2_swin_transformer_1_window_attention_1_dense_6_kernel_read_readvariableopQ
Msavev2_swin_transformer_1_window_attention_1_dense_6_bias_read_readvariableopM
Isavev2_swin_transformer_1_layer_normalization_3_gamma_read_readvariableopL
Hsavev2_swin_transformer_1_layer_normalization_3_beta_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableopM
Isavev2_swin_transformer_1_window_attention_1_variable_read_readvariableop	;
7savev2_patch_merging_dense_9_kernel_read_readvariableop)
%savev2_statevar_1_read_readvariableop	'
#savev2_statevar_read_readvariableop	&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B9layer_with_weights-2/attn_mask/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEBJlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_swin_transformer_1_variable_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop7savev2_patch_embedding_dense_kernel_read_readvariableop5savev2_patch_embedding_dense_bias_read_readvariableop?savev2_patch_embedding_embedding_embeddings_read_readvariableopEsavev2_swin_transformer_layer_normalization_gamma_read_readvariableopDsavev2_swin_transformer_layer_normalization_beta_read_readvariableopCsavev2_swin_transformer_window_attention_weight_read_readvariableopKsavev2_swin_transformer_window_attention_dense_1_kernel_read_readvariableopIsavev2_swin_transformer_window_attention_dense_1_bias_read_readvariableopKsavev2_swin_transformer_window_attention_dense_2_kernel_read_readvariableopIsavev2_swin_transformer_window_attention_dense_2_bias_read_readvariableopGsavev2_swin_transformer_layer_normalization_1_gamma_read_readvariableopFsavev2_swin_transformer_layer_normalization_1_beta_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableopEsavev2_swin_transformer_window_attention_variable_read_readvariableopIsavev2_swin_transformer_1_layer_normalization_2_gamma_read_readvariableopHsavev2_swin_transformer_1_layer_normalization_2_beta_read_readvariableopGsavev2_swin_transformer_1_window_attention_1_weight_read_readvariableopOsavev2_swin_transformer_1_window_attention_1_dense_5_kernel_read_readvariableopMsavev2_swin_transformer_1_window_attention_1_dense_5_bias_read_readvariableopOsavev2_swin_transformer_1_window_attention_1_dense_6_kernel_read_readvariableopMsavev2_swin_transformer_1_window_attention_1_dense_6_bias_read_readvariableopIsavev2_swin_transformer_1_layer_normalization_3_gamma_read_readvariableopHsavev2_swin_transformer_1_layer_normalization_3_beta_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableopIsavev2_swin_transformer_1_window_attention_1_variable_read_readvariableop7savev2_patch_merging_dense_9_kernel_read_readvariableop%savev2_statevar_1_read_readvariableop#savev2_statevar_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,				?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:	?W:W:@:@:	?@:@:@:	:	@?:?:@@:@:@:@:	@?:?:	?@:@::@:@:	:	@?:?:@@:@:@:@:	@?:?:	?@:@::
??::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
:?:%!

_output_shapes
:	?W: 

_output_shapes
:W:$ 

_output_shapes

:@: 

_output_shapes
:@:%!

_output_shapes
:	?@: 

_output_shapes
:@: 

_output_shapes
:@:$	 

_output_shapes

:	:%
!

_output_shapes
:	@?:!

_output_shapes	
:?:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:	:%!

_output_shapes
:	@?:!

_output_shapes	
:?:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:% !

_output_shapes
:	?@: !

_output_shapes
:@:$" 

_output_shapes

::&#"
 
_output_shapes
:
??: $

_output_shapes
:: %

_output_shapes
::&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
?

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_13304

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q???j
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????u
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????o
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:???????????_
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
1__inference_swin_transformer_1_layer_call_fn_1176
x
unknown:@
	unknown_0:@
	unknown_1:	@?
	unknown_2:	?
	unknown_3:	
	unknown_4:	 
	unknown_5:?
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:	@?

unknown_11:	?

unknown_12:	?@

unknown_13:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_1156`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:??????????@: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????@

_user_specified_namex
??
?
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2631
xG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@M
:window_attention_dense_1_tensordot_readvariableop_resource:	@?G
8window_attention_dense_1_biasadd_readvariableop_resource:	?D
2window_attention_reshape_1_readvariableop_resource:	2
 window_attention_gather_resource:	L
:window_attention_dense_2_tensordot_readvariableop_resource:@@F
8window_attention_dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@G
4sequential_dense_3_tensordot_readvariableop_resource:	@?A
2sequential_dense_3_biasadd_readvariableop_resource:	?G
4sequential_dense_4_tensordot_readvariableop_resource:	?@@
2sequential_dense_4_biasadd_readvariableop_resource:@
identity??,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?)sequential/dense_3/BiasAdd/ReadVariableOp?+sequential/dense_3/Tensordot/ReadVariableOp?)sequential/dense_4/BiasAdd/ReadVariableOp?+sequential/dense_4/Tensordot/ReadVariableOp?window_attention/Gather?)window_attention/Reshape_1/ReadVariableOp?/window_attention/dense_1/BiasAdd/ReadVariableOp?1window_attention/dense_1/Tensordot/ReadVariableOp?/window_attention/dense_2/BiasAdd/ReadVariableOp?1window_attention/dense_2/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanx;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:???????????
-layer_normalization/moments/SquaredDifferenceSquaredDifferencex1layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/mul_1Mulx%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   ?
ReshapeReshape'layer_normalization/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:?????????  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_1ReshapeReshape:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!?????????@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention/dense_1/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0q
'window_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'window_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       j
(window_attention/dense_1/Tensordot/ShapeShapeReshape_3:output:0*
T0*
_output_shapes
:r
0window_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_1/Tensordot/GatherV2GatherV21window_attention/dense_1/Tensordot/Shape:output:00window_attention/dense_1/Tensordot/free:output:09window_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2window_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention/dense_1/Tensordot/GatherV2_1GatherV21window_attention/dense_1/Tensordot/Shape:output:00window_attention/dense_1/Tensordot/axes:output:0;window_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(window_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
'window_attention/dense_1/Tensordot/ProdProd4window_attention/dense_1/Tensordot/GatherV2:output:01window_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention/dense_1/Tensordot/Prod_1Prod6window_attention/dense_1/Tensordot/GatherV2_1:output:03window_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)window_attention/dense_1/Tensordot/concatConcatV20window_attention/dense_1/Tensordot/free:output:00window_attention/dense_1/Tensordot/axes:output:07window_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(window_attention/dense_1/Tensordot/stackPack0window_attention/dense_1/Tensordot/Prod:output:02window_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
,window_attention/dense_1/Tensordot/transpose	TransposeReshape_3:output:02window_attention/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
*window_attention/dense_1/Tensordot/ReshapeReshape0window_attention/dense_1/Tensordot/transpose:y:01window_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
)window_attention/dense_1/Tensordot/MatMulMatMul3window_attention/dense_1/Tensordot/Reshape:output:09window_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????u
*window_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?r
0window_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_1/Tensordot/concat_1ConcatV24window_attention/dense_1/Tensordot/GatherV2:output:03window_attention/dense_1/Tensordot/Const_2:output:09window_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
"window_attention/dense_1/TensordotReshape3window_attention/dense_1/Tensordot/MatMul:product:04window_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:???????????
/window_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
 window_attention/dense_1/BiasAddBiasAdd+window_attention/dense_1/Tensordot:output:07window_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????{
window_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention/ReshapeReshape)window_attention/dense_1/BiasAdd:output:0'window_attention/Reshape/shape:output:0*
T0*3
_output_shapes!
:?????????|
window_attention/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
window_attention/transpose	Transpose!window_attention/Reshape:output:0(window_attention/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????n
$window_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&window_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&window_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
window_attention/strided_sliceStridedSlicewindow_attention/transpose:y:0-window_attention/strided_slice/stack:output:0/window_attention/strided_slice/stack_1:output:0/window_attention/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskp
&window_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention/strided_slice_1StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_1/stack:output:01window_attention/strided_slice_1/stack_1:output:01window_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskp
&window_attention/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention/strided_slice_2StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_2/stack:output:01window_attention/strided_slice_2/stack_1:output:01window_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_mask[
window_attention/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
window_attention/mulMul'window_attention/strided_slice:output:0window_attention/mul/y:output:0*
T0*/
_output_shapes
:?????????z
!window_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention/transpose_1	Transpose)window_attention/strided_slice_1:output:0*window_attention/transpose_1/perm:output:0*
T0*/
_output_shapes
:??????????
window_attention/matmulBatchMatMulV2window_attention/mul:z:0 window_attention/transpose_1:y:0*
T0*/
_output_shapes
:??????????
)window_attention/Reshape_1/ReadVariableOpReadVariableOp2window_attention_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	s
 window_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
window_attention/Reshape_1Reshape1window_attention/Reshape_1/ReadVariableOp:value:0)window_attention/Reshape_1/shape:output:0*
T0	*
_output_shapes
:?
window_attention/GatherResourceGather window_attention_gather_resource#window_attention/Reshape_1:output:0*
Tindices0	*
_output_shapes

:*
dtype0p
window_attention/IdentityIdentity window_attention/Gather:output:0*
T0*
_output_shapes

:u
 window_attention/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?????
window_attention/Reshape_2Reshape"window_attention/Identity:output:0)window_attention/Reshape_2/shape:output:0*
T0*"
_output_shapes
:v
!window_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
window_attention/transpose_2	Transpose#window_attention/Reshape_2:output:0*window_attention/transpose_2/perm:output:0*
T0*"
_output_shapes
:a
window_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention/ExpandDims
ExpandDims window_attention/transpose_2:y:0(window_attention/ExpandDims/dim:output:0*
T0*&
_output_shapes
:?
window_attention/addAddV2 window_attention/matmul:output:0$window_attention/ExpandDims:output:0*
T0*/
_output_shapes
:?????????w
window_attention/SoftmaxSoftmaxwindow_attention/add:z:0*
T0*/
_output_shapes
:?????????k
&window_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
$window_attention/dropout/dropout/MulMul"window_attention/Softmax:softmax:0/window_attention/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????x
&window_attention/dropout/dropout/ShapeShape"window_attention/Softmax:softmax:0*
T0*
_output_shapes
:?
=window_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform/window_attention/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0t
/window_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
-window_attention/dropout/dropout/GreaterEqualGreaterEqualFwindow_attention/dropout/dropout/random_uniform/RandomUniform:output:08window_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
%window_attention/dropout/dropout/CastCast1window_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
&window_attention/dropout/dropout/Mul_1Mul(window_attention/dropout/dropout/Mul:z:0)window_attention/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
window_attention/matmul_1BatchMatMulV2*window_attention/dropout/dropout/Mul_1:z:0)window_attention/strided_slice_2:output:0*
T0*/
_output_shapes
:?????????z
!window_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention/transpose_3	Transpose"window_attention/matmul_1:output:0*window_attention/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????u
 window_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   ?
window_attention/Reshape_3Reshape window_attention/transpose_3:y:0)window_attention/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention/dense_2/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0q
'window_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'window_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
(window_attention/dense_2/Tensordot/ShapeShape#window_attention/Reshape_3:output:0*
T0*
_output_shapes
:r
0window_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_2/Tensordot/GatherV2GatherV21window_attention/dense_2/Tensordot/Shape:output:00window_attention/dense_2/Tensordot/free:output:09window_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2window_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention/dense_2/Tensordot/GatherV2_1GatherV21window_attention/dense_2/Tensordot/Shape:output:00window_attention/dense_2/Tensordot/axes:output:0;window_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(window_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
'window_attention/dense_2/Tensordot/ProdProd4window_attention/dense_2/Tensordot/GatherV2:output:01window_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention/dense_2/Tensordot/Prod_1Prod6window_attention/dense_2/Tensordot/GatherV2_1:output:03window_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)window_attention/dense_2/Tensordot/concatConcatV20window_attention/dense_2/Tensordot/free:output:00window_attention/dense_2/Tensordot/axes:output:07window_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(window_attention/dense_2/Tensordot/stackPack0window_attention/dense_2/Tensordot/Prod:output:02window_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
,window_attention/dense_2/Tensordot/transpose	Transpose#window_attention/Reshape_3:output:02window_attention/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
*window_attention/dense_2/Tensordot/ReshapeReshape0window_attention/dense_2/Tensordot/transpose:y:01window_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
)window_attention/dense_2/Tensordot/MatMulMatMul3window_attention/dense_2/Tensordot/Reshape:output:09window_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@t
*window_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@r
0window_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_2/Tensordot/concat_1ConcatV24window_attention/dense_2/Tensordot/GatherV2:output:03window_attention/dense_2/Tensordot/Const_2:output:09window_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
"window_attention/dense_2/TensordotReshape3window_attention/dense_2/Tensordot/MatMul:product:04window_attention/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@?
/window_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
 window_attention/dense_2/BiasAddBiasAdd+window_attention/dense_2/Tensordot:output:07window_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@m
(window_attention/dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
&window_attention/dropout/dropout_1/MulMul)window_attention/dense_2/BiasAdd:output:01window_attention/dropout/dropout_1/Const:output:0*
T0*+
_output_shapes
:?????????@?
(window_attention/dropout/dropout_1/ShapeShape)window_attention/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:?
?window_attention/dropout/dropout_1/random_uniform/RandomUniformRandomUniform1window_attention/dropout/dropout_1/Shape:output:0*
T0*+
_output_shapes
:?????????@*
dtype0v
1window_attention/dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
/window_attention/dropout/dropout_1/GreaterEqualGreaterEqualHwindow_attention/dropout/dropout_1/random_uniform/RandomUniform:output:0:window_attention/dropout/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????@?
'window_attention/dropout/dropout_1/CastCast3window_attention/dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????@?
(window_attention/dropout/dropout_1/Mul_1Mul*window_attention/dropout/dropout_1/Mul:z:0+window_attention/dropout/dropout_1/Cast:y:0*
T0*+
_output_shapes
:?????????@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   ?
	Reshape_4Reshape,window_attention/dropout/dropout_1/Mul_1:z:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:?????????@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!?????????@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:?????????  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   y
	Reshape_7ReshapeReshape_6:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:??????????@Q
drop_path/ShapeShapeReshape_7:output:0*
T0*
_output_shapes
:g
drop_path/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
drop_path/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
drop_path/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path/strided_sliceStridedSlicedrop_path/Shape:output:0&drop_path/strided_slice/stack:output:0(drop_path/strided_slice/stack_1:output:0(drop_path/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 drop_path/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :b
 drop_path/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
drop_path/random_uniform/shapePack drop_path/strided_slice:output:0)drop_path/random_uniform/shape/1:output:0)drop_path/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:?
&drop_path/random_uniform/RandomUniformRandomUniform'drop_path/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0T
drop_path/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/addAddV2drop_path/add/x:output:0/drop_path/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:?????????a
drop_path/FloorFloordrop_path/add:z:0*
T0*+
_output_shapes
:?????????X
drop_path/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/truedivRealDivReshape_7:output:0drop_path/truediv/y:output:0*
T0*,
_output_shapes
:??????????@w
drop_path/mulMuldrop_path/truediv:z:0drop_path/Floor:y:0*
T0*,
_output_shapes
:??????????@Y
addAddV2xdrop_path/mul:z:0*
T0*,
_output_shapes
:??????????@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@?
+sequential/dense_3/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0k
!sequential/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
"sequential/dense_3/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_3/Tensordot/GatherV2GatherV2+sequential/dense_3/Tensordot/Shape:output:0*sequential/dense_3/Tensordot/free:output:03sequential/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential/dense_3/Tensordot/GatherV2_1GatherV2+sequential/dense_3/Tensordot/Shape:output:0*sequential/dense_3/Tensordot/axes:output:05sequential/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!sequential/dense_3/Tensordot/ProdProd.sequential/dense_3/Tensordot/GatherV2:output:0+sequential/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
#sequential/dense_3/Tensordot/Prod_1Prod0sequential/dense_3/Tensordot/GatherV2_1:output:0-sequential/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/dense_3/Tensordot/concatConcatV2*sequential/dense_3/Tensordot/free:output:0*sequential/dense_3/Tensordot/axes:output:01sequential/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"sequential/dense_3/Tensordot/stackPack*sequential/dense_3/Tensordot/Prod:output:0,sequential/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
&sequential/dense_3/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0,sequential/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
$sequential/dense_3/Tensordot/ReshapeReshape*sequential/dense_3/Tensordot/transpose:y:0+sequential/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
#sequential/dense_3/Tensordot/MatMulMatMul-sequential/dense_3/Tensordot/Reshape:output:03sequential/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
$sequential/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?l
*sequential/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_3/Tensordot/concat_1ConcatV2.sequential/dense_3/Tensordot/GatherV2:output:0-sequential/dense_3/Tensordot/Const_2:output:03sequential/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential/dense_3/TensordotReshape-sequential/dense_3/Tensordot/MatMul:product:0.sequential/dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/dense_3/BiasAddBiasAdd%sequential/dense_3/Tensordot:output:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????e
 sequential/activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
sequential/activation/Gelu/mulMul)sequential/activation/Gelu/mul/x:output:0#sequential/dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:???????????f
!sequential/activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
"sequential/activation/Gelu/truedivRealDiv#sequential/dense_3/BiasAdd:output:0*sequential/activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:????????????
sequential/activation/Gelu/ErfErf&sequential/activation/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????e
 sequential/activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sequential/activation/Gelu/addAddV2)sequential/activation/Gelu/add/x:output:0"sequential/activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
 sequential/activation/Gelu/mul_1Mul"sequential/activation/Gelu/mul:z:0"sequential/activation/Gelu/add:z:0*
T0*-
_output_shapes
:???????????g
"sequential/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
 sequential/dropout_1/dropout/MulMul$sequential/activation/Gelu/mul_1:z:0+sequential/dropout_1/dropout/Const:output:0*
T0*-
_output_shapes
:???????????v
"sequential/dropout_1/dropout/ShapeShape$sequential/activation/Gelu/mul_1:z:0*
T0*
_output_shapes
:?
9sequential/dropout_1/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_1/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0p
+sequential/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
)sequential/dropout_1/dropout/GreaterEqualGreaterEqualBsequential/dropout_1/dropout/random_uniform/RandomUniform:output:04sequential/dropout_1/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
!sequential/dropout_1/dropout/CastCast-sequential/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
"sequential/dropout_1/dropout/Mul_1Mul$sequential/dropout_1/dropout/Mul:z:0%sequential/dropout_1/dropout/Cast:y:0*
T0*-
_output_shapes
:????????????
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0k
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       x
"sequential/dense_4/Tensordot/ShapeShape&sequential/dropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
&sequential/dense_4/Tensordot/transpose	Transpose&sequential/dropout_1/dropout/Mul_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@g
"sequential/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
 sequential/dropout_2/dropout/MulMul#sequential/dense_4/BiasAdd:output:0+sequential/dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:??????????@u
"sequential/dropout_2/dropout/ShapeShape#sequential/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:?
9sequential/dropout_2/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype0p
+sequential/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
)sequential/dropout_2/dropout/GreaterEqualGreaterEqualBsequential/dropout_2/dropout/random_uniform/RandomUniform:output:04sequential/dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@?
!sequential/dropout_2/dropout/CastCast-sequential/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@?
"sequential/dropout_2/dropout/Mul_1Mul$sequential/dropout_2/dropout/Mul:z:0%sequential/dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@g
drop_path/Shape_1Shape&sequential/dropout_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:i
drop_path/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!drop_path/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!drop_path/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path/strided_slice_1StridedSlicedrop_path/Shape_1:output:0(drop_path/strided_slice_1/stack:output:0*drop_path/strided_slice_1/stack_1:output:0*drop_path/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"drop_path/random_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d
"drop_path/random_uniform_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
 drop_path/random_uniform_1/shapePack"drop_path/strided_slice_1:output:0+drop_path/random_uniform_1/shape/1:output:0+drop_path/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
(drop_path/random_uniform_1/RandomUniformRandomUniform)drop_path/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0V
drop_path/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/add_1AddV2drop_path/add_1/x:output:01drop_path/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:?????????e
drop_path/Floor_1Floordrop_path/add_1:z:0*
T0*+
_output_shapes
:?????????Z
drop_path/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/truediv_1RealDiv&sequential/dropout_2/dropout/Mul_1:z:0drop_path/truediv_1/y:output:0*
T0*,
_output_shapes
:??????????@}
drop_path/mul_1Muldrop_path/truediv_1:z:0drop_path/Floor_1:y:0*
T0*,
_output_shapes
:??????????@c
add_1AddV2add:z:0drop_path/mul_1:z:0*
T0*,
_output_shapes
:??????????@?
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp,^sequential/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp^window_attention/Gather*^window_attention/Reshape_1/ReadVariableOp0^window_attention/dense_1/BiasAdd/ReadVariableOp2^window_attention/dense_1/Tensordot/ReadVariableOp0^window_attention/dense_2/BiasAdd/ReadVariableOp2^window_attention/dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????@: : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2Z
+sequential/dense_3/Tensordot/ReadVariableOp+sequential/dense_3/Tensordot/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp22
window_attention/Gatherwindow_attention/Gather2V
)window_attention/Reshape_1/ReadVariableOp)window_attention/Reshape_1/ReadVariableOp2b
/window_attention/dense_1/BiasAdd/ReadVariableOp/window_attention/dense_1/BiasAdd/ReadVariableOp2f
1window_attention/dense_1/Tensordot/ReadVariableOp1window_attention/dense_1/Tensordot/ReadVariableOp2b
/window_attention/dense_2/BiasAdd/ReadVariableOp/window_attention/dense_2/BiasAdd/ReadVariableOp2f
1window_attention/dense_2/Tensordot/ReadVariableOp1window_attention/dense_2/Tensordot/ReadVariableOp:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_13143

inputs

identity_1T
IdentityIdentityinputs*
T0*-
_output_shapes
:???????????a

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_random_crop_layer_call_and_return_conditional_losses_12000

inputs
cond_input_1:	
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
sub/yConst*
_output_shapes
: *
dtype0*
value	B :@S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :@Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: g
Rank/packedPackGreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:e
	All/inputPackGreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0inputscond_input_1*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_11855*.
output_shapes
:?????????@@*"
then_branchR
cond_true_11854b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:?????????@@m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:?????????@@M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: 2
condcond:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_11206

inputs
unknown:@
	unknown_0:@
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:	@?
	unknown_5:	?
	unknown_6:	
	unknown_7:	
	unknown_8:@@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:	@?

unknown_13:	?

unknown_14:	?@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@?

unknown_19:	?

unknown_20:	

unknown_21:	!

unknown_22:?

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:	@?

unknown_28:	?

unknown_29:	?@

unknown_30:@

unknown_31:
??

unknown_32:	?W

unknown_33:W
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*E
_read_only_resource_inputs'
%#	
 !"#*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
G
+__inference_random_crop_layer_call_fn_11774

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10007h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
T
8__inference_global_average_pooling1d_layer_call_fn_12692

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_11131
input_1
unknown:@
	unknown_0:@
	unknown_1:	?@
	unknown_2:@
	unknown_3:@
	unknown_4:	@?
	unknown_5:	?
	unknown_6:	
	unknown_7:	
	unknown_8:@@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:	@?

unknown_13:	?

unknown_14:	?@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@?

unknown_19:	?

unknown_20:	

unknown_21:	!

unknown_22:?

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:	@?

unknown_28:	?

unknown_29:	?@

unknown_30:@

unknown_31:
??

unknown_32:	?W

unknown_33:W
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*E
_read_only_resource_inputs'
%#	
 !"#*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_9941o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
}
,__inference_patch_merging_layer_call_fn_2744
x
unknown:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_patch_merging_layer_call_and_return_conditional_losses_2738`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????@: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:??????????@

_user_specified_namex
?

?
C__inference_dense_10_layer_call_and_return_conditional_losses_10098

inputs1
matmul_readvariableop_resource:	?W-
biasadd_readvariableop_resource:W
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?W*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Wr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:W*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????WV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????W`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????Ww
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_12511

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
5map_while_stateless_random_flip_left_right_true_10271v
rmap_while_stateless_random_flip_left_right_reversev2_map_while_stateless_random_flip_left_right_control_dependency7
3map_while_stateless_random_flip_left_right_identity?
9map/while/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:?
4map/while/stateless_random_flip_left_right/ReverseV2	ReverseV2rmap_while_stateless_random_flip_left_right_reversev2_map_while_stateless_random_flip_left_right_control_dependencyBmap/while/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*"
_output_shapes
:@@?
3map/while/stateless_random_flip_left_right/IdentityIdentity=map/while/stateless_random_flip_left_right/ReverseV2:output:0*
T0*"
_output_shapes
:@@"s
3map_while_stateless_random_flip_left_right_identity<map/while/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:@@:( $
"
_output_shapes
:@@
̇
?
random_crop_cond_true_11438!
random_crop_cond_shape_inputsG
9random_crop_cond_stateful_uniform_rngreadandskip_resource:	
random_crop_cond_identity??3random_crop/cond/crop_to_bounding_box/Assert/Assert?5random_crop/cond/crop_to_bounding_box/Assert_1/Assert?5random_crop/cond/crop_to_bounding_box/Assert_2/Assert?5random_crop/cond/crop_to_bounding_box/Assert_3/Assert?0random_crop/cond/stateful_uniform/RngReadAndSkipc
random_crop/cond/ShapeShaperandom_crop_cond_shape_inputs*
T0*
_output_shapes
:w
$random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????y
&random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????p
&random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/cond/strided_sliceStridedSlicerandom_crop/cond/Shape:output:0-random_crop/cond/strided_slice/stack:output:0/random_crop/cond/strided_slice/stack_1:output:0/random_crop/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
random_crop/cond/sub/yConst*
_output_shapes
: *
dtype0*
value	B :@?
random_crop/cond/subSub'random_crop/cond/strided_slice:output:0random_crop/cond/sub/y:output:0*
T0*
_output_shapes
: y
&random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
(random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop/cond/strided_slice_1StridedSlicerandom_crop/cond/Shape:output:0/random_crop/cond/strided_slice_1/stack:output:01random_crop/cond/strided_slice_1/stack_1:output:01random_crop/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
random_crop/cond/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :@?
random_crop/cond/sub_1Sub)random_crop/cond/strided_slice_1:output:0!random_crop/cond/sub_1/y:output:0*
T0*
_output_shapes
: q
'random_crop/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
%random_crop/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : k
%random_crop/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????q
'random_crop/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
&random_crop/cond/stateful_uniform/ProdProd0random_crop/cond/stateful_uniform/shape:output:00random_crop/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: j
(random_crop/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
(random_crop/cond/stateful_uniform/Cast_1Cast/random_crop/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
0random_crop/cond/stateful_uniform/RngReadAndSkipRngReadAndSkip9random_crop_cond_stateful_uniform_rngreadandskip_resource1random_crop/cond/stateful_uniform/Cast/x:output:0,random_crop/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:
5random_crop/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7random_crop/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7random_crop/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/random_crop/cond/stateful_uniform/strided_sliceStridedSlice8random_crop/cond/stateful_uniform/RngReadAndSkip:value:0>random_crop/cond/stateful_uniform/strided_slice/stack:output:0@random_crop/cond/stateful_uniform/strided_slice/stack_1:output:0@random_crop/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
)random_crop/cond/stateful_uniform/BitcastBitcast8random_crop/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
7random_crop/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9random_crop/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9random_crop/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1random_crop/cond/stateful_uniform/strided_slice_1StridedSlice8random_crop/cond/stateful_uniform/RngReadAndSkip:value:0@random_crop/cond/stateful_uniform/strided_slice_1/stack:output:0Brandom_crop/cond/stateful_uniform/strided_slice_1/stack_1:output:0Brandom_crop/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
+random_crop/cond/stateful_uniform/Bitcast_1Bitcast:random_crop/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0g
%random_crop/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
!random_crop/cond/stateful_uniformStatelessRandomUniformIntV20random_crop/cond/stateful_uniform/shape:output:04random_crop/cond/stateful_uniform/Bitcast_1:output:02random_crop/cond/stateful_uniform/Bitcast:output:0.random_crop/cond/stateful_uniform/alg:output:0.random_crop/cond/stateful_uniform/min:output:0.random_crop/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0p
&random_crop/cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(random_crop/cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(random_crop/cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop/cond/strided_slice_2StridedSlice*random_crop/cond/stateful_uniform:output:0/random_crop/cond/strided_slice_2/stack:output:01random_crop/cond/strided_slice_2/stack_1:output:01random_crop/cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
random_crop/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :y
random_crop/cond/addAddV2random_crop/cond/sub:z:0random_crop/cond/add/y:output:0*
T0*
_output_shapes
: ?
random_crop/cond/modFloorMod)random_crop/cond/strided_slice_2:output:0random_crop/cond/add:z:0*
T0*
_output_shapes
: p
&random_crop/cond/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(random_crop/cond/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(random_crop/cond/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop/cond/strided_slice_3StridedSlice*random_crop/cond/stateful_uniform:output:0/random_crop/cond/strided_slice_3/stack:output:01random_crop/cond/strided_slice_3/stack_1:output:01random_crop/cond/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
random_crop/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
random_crop/cond/add_1AddV2random_crop/cond/sub_1:z:0!random_crop/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
random_crop/cond/mod_1FloorMod)random_crop/cond/strided_slice_3:output:0random_crop/cond/add_1:z:0*
T0*
_output_shapes
: x
+random_crop/cond/crop_to_bounding_box/ShapeShaperandom_crop_cond_shape_inputs*
T0*
_output_shapes
:?
-random_crop/cond/crop_to_bounding_box/unstackUnpack4random_crop/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numv
4random_crop/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
2random_crop/cond/crop_to_bounding_box/GreaterEqualGreaterEqualrandom_crop/cond/mod_1:z:0=random_crop/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
2random_crop/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
:random_crop/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
3random_crop/cond/crop_to_bounding_box/Assert/AssertAssert6random_crop/cond/crop_to_bounding_box/GreaterEqual:z:0Crandom_crop/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 x
6random_crop/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
4random_crop/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualrandom_crop/cond/mod:z:0?random_crop/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
4random_crop/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
<random_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
5random_crop/cond/crop_to_bounding_box/Assert_1/AssertAssert8random_crop/cond/crop_to_bounding_box/GreaterEqual_1:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:04^random_crop/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 m
+random_crop/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B :@?
)random_crop/cond/crop_to_bounding_box/addAddV24random_crop/cond/crop_to_bounding_box/add/x:output:0random_crop/cond/mod_1:z:0*
T0*
_output_shapes
: s
1random_crop/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B :@?
/random_crop/cond/crop_to_bounding_box/LessEqual	LessEqual-random_crop/cond/crop_to_bounding_box/add:z:0:random_crop/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
4random_crop/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
<random_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
5random_crop/cond/crop_to_bounding_box/Assert_2/AssertAssert3random_crop/cond/crop_to_bounding_box/LessEqual:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:06^random_crop/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 o
-random_crop/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B :@?
+random_crop/cond/crop_to_bounding_box/add_1AddV26random_crop/cond/crop_to_bounding_box/add_1/x:output:0random_crop/cond/mod:z:0*
T0*
_output_shapes
: u
3random_crop/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B :@?
1random_crop/cond/crop_to_bounding_box/LessEqual_1	LessEqual/random_crop/cond/crop_to_bounding_box/add_1:z:0<random_crop/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
4random_crop/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
<random_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
5random_crop/cond/crop_to_bounding_box/Assert_3/AssertAssert5random_crop/cond/crop_to_bounding_box/LessEqual_1:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:06^random_crop/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
8random_crop/cond/crop_to_bounding_box/control_dependencyIdentityrandom_crop_cond_shape_inputs4^random_crop/cond/crop_to_bounding_box/Assert/Assert6^random_crop/cond/crop_to_bounding_box/Assert_1/Assert6^random_crop/cond/crop_to_bounding_box/Assert_2/Assert6^random_crop/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????@@o
-random_crop/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : o
-random_crop/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
+random_crop/cond/crop_to_bounding_box/stackPack6random_crop/cond/crop_to_bounding_box/stack/0:output:0random_crop/cond/mod:z:0random_crop/cond/mod_1:z:06random_crop/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
-random_crop/cond/crop_to_bounding_box/Shape_1ShapeArandom_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
9random_crop/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;random_crop/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;random_crop/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3random_crop/cond/crop_to_bounding_box/strided_sliceStridedSlice6random_crop/cond/crop_to_bounding_box/Shape_1:output:0Brandom_crop/cond/crop_to_bounding_box/strided_slice/stack:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
-random_crop/cond/crop_to_bounding_box/Shape_2ShapeArandom_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
;random_crop/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
=random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5random_crop/cond/crop_to_bounding_box/strided_slice_1StridedSlice6random_crop/cond/crop_to_bounding_box/Shape_2:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Frandom_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Frandom_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/random_crop/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B :@q
/random_crop/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B :@?
-random_crop/cond/crop_to_bounding_box/stack_1Pack<random_crop/cond/crop_to_bounding_box/strided_slice:output:08random_crop/cond/crop_to_bounding_box/stack_1/1:output:08random_crop/cond/crop_to_bounding_box/stack_1/2:output:0>random_crop/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
+random_crop/cond/crop_to_bounding_box/SliceSliceArandom_crop/cond/crop_to_bounding_box/control_dependency:output:04random_crop/cond/crop_to_bounding_box/stack:output:06random_crop/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:?????????@@?
random_crop/cond/IdentityIdentity4random_crop/cond/crop_to_bounding_box/Slice:output:0^random_crop/cond/NoOp*
T0*/
_output_shapes
:?????????@@?
random_crop/cond/NoOpNoOp4^random_crop/cond/crop_to_bounding_box/Assert/Assert6^random_crop/cond/crop_to_bounding_box/Assert_1/Assert6^random_crop/cond/crop_to_bounding_box/Assert_2/Assert6^random_crop/cond/crop_to_bounding_box/Assert_3/Assert1^random_crop/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "?
random_crop_cond_identity"random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: 2j
3random_crop/cond/crop_to_bounding_box/Assert/Assert3random_crop/cond/crop_to_bounding_box/Assert/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_1/Assert5random_crop/cond/crop_to_bounding_box/Assert_1/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_2/Assert5random_crop/cond/crop_to_bounding_box/Assert_2/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_3/Assert5random_crop/cond/crop_to_bounding_box/Assert_3/Assert2d
0random_crop/cond/stateful_uniform/RngReadAndSkip0random_crop/cond/stateful_uniform/RngReadAndSkip:5 1
/
_output_shapes
:?????????@@
?	
?
Arandom_flip_map_while_stateless_random_flip_left_right_true_11655?
?random_flip_map_while_stateless_random_flip_left_right_reversev2_random_flip_map_while_stateless_random_flip_left_right_control_dependencyC
?random_flip_map_while_stateless_random_flip_left_right_identity?
Erandom_flip/map/while/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:?
@random_flip/map/while/stateless_random_flip_left_right/ReverseV2	ReverseV2?random_flip_map_while_stateless_random_flip_left_right_reversev2_random_flip_map_while_stateless_random_flip_left_right_control_dependencyNrandom_flip/map/while/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*"
_output_shapes
:@@?
?random_flip/map/while/stateless_random_flip_left_right/IdentityIdentityIrandom_flip/map/while/stateless_random_flip_left_right/ReverseV2:output:0*
T0*"
_output_shapes
:@@"?
?random_flip_map_while_stateless_random_flip_left_right_identityHrandom_flip/map/while/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:@@:( $
"
_output_shapes
:@@
?6
?
@__inference_model_layer_call_and_return_conditional_losses_10967
input_1'
patch_embedding_10890:@#
patch_embedding_10892:@(
patch_embedding_10894:	?@$
swin_transformer_10897:@$
swin_transformer_10899:@)
swin_transformer_10901:	@?%
swin_transformer_10903:	?(
swin_transformer_10905:	(
swin_transformer_10907:	(
swin_transformer_10909:@@$
swin_transformer_10911:@$
swin_transformer_10913:@$
swin_transformer_10915:@)
swin_transformer_10917:	@?%
swin_transformer_10919:	?)
swin_transformer_10921:	?@$
swin_transformer_10923:@&
swin_transformer_1_10926:@&
swin_transformer_1_10928:@+
swin_transformer_1_10930:	@?'
swin_transformer_1_10932:	?*
swin_transformer_1_10934:	*
swin_transformer_1_10936:	/
swin_transformer_1_10938:?*
swin_transformer_1_10940:@@&
swin_transformer_1_10942:@&
swin_transformer_1_10944:@&
swin_transformer_1_10946:@+
swin_transformer_1_10948:	@?'
swin_transformer_1_10950:	?+
swin_transformer_1_10952:	?@&
swin_transformer_1_10954:@'
patch_merging_10957:
??!
dense_10_10961:	?W
dense_10_10963:W
identity?? dense_10/StatefulPartitionedCall?'patch_embedding/StatefulPartitionedCall?%patch_merging/StatefulPartitionedCall?(swin_transformer/StatefulPartitionedCall?*swin_transformer_1/StatefulPartitionedCall?
random_crop/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10007?
random_flip/PartitionedCallPartitionedCall$random_crop/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10013?
patch_extract/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9773?
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_10890patch_embedding_10892patch_embedding_10894*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9785?
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_10897swin_transformer_10899swin_transformer_10901swin_transformer_10903swin_transformer_10905swin_transformer_10907swin_transformer_10909swin_transformer_10911swin_transformer_10913swin_transformer_10915swin_transformer_10917swin_transformer_10919swin_transformer_10921swin_transformer_10923*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9825?
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_10926swin_transformer_1_10928swin_transformer_1_10930swin_transformer_1_10932swin_transformer_1_10934swin_transformer_1_10936swin_transformer_1_10938swin_transformer_1_10940swin_transformer_1_10942swin_transformer_1_10944swin_transformer_1_10946swin_transformer_1_10948swin_transformer_1_10950swin_transformer_1_10952swin_transformer_1_10954*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9889?
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_10957*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9927?
(global_average_pooling1d/PartitionedCallPartitionedCall.patch_merging/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_10_10961dense_10_10963*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_10098x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????W?
NoOpNoOp!^dense_10/StatefulPartitionedCall(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?
?
B__inference_dense_7_layer_call_and_return_conditional_losses_12443

inputs4
!tensordot_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????e
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:???????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_12244
dense_3_input
unknown:	@?
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_3_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12233t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:??????????@
'
_user_specified_namedense_3_input
?
?
'__inference_dense_3_layer_call_fn_13081

inputs
unknown:	@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_12162u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?"
b
F__inference_random_crop_layer_call_and_return_conditional_losses_10007

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :@U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: E
CastCastmul:z:0*

DstT0*

SrcT0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?BQ
truedivRealDivCast:y:0truediv/y:output:0*
T0*
_output_shapes
: K
Cast_1Casttruediv:z:0*

DstT0*

SrcT0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :@W
mul_1Mulstrided_slice:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
Cast_2Cast	mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?BW
	truediv_1RealDiv
Cast_2:y:0truediv_1/y:output:0*
T0*
_output_shapes
: M
Cast_3Casttruediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: W
MinimumMinimumstrided_slice:output:0
Cast_1:y:0*
T0*
_output_shapes
: [
	Minimum_1Minimumstrided_slice_1:output:0
Cast_3:y:0*
T0*
_output_shapes
: P
subSubstrided_slice:output:0Minimum:z:0*
T0*
_output_shapes
: G
Cast_4Castsub:z:0*

DstT0*

SrcT0*
_output_shapes
: P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @W
	truediv_2RealDiv
Cast_4:y:0truediv_2/y:output:0*
T0*
_output_shapes
: M
Cast_5Casttruediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: V
sub_1Substrided_slice_1:output:0Minimum_1:z:0*
T0*
_output_shapes
: I
Cast_6Cast	sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @W
	truediv_3RealDiv
Cast_6:y:0truediv_3/y:output:0*
T0*
_output_shapes
: M
Cast_7Casttruediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: I
stack/0Const*
_output_shapes
: *
dtype0*
value	B : I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : w
stackPackstack/0:output:0
Cast_5:y:0
Cast_7:y:0stack/3:output:0*
N*
T0*
_output_shapes
:T
	stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????T
	stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
stack_1Packstack_1/0:output:0Minimum:z:0Minimum_1:z:0stack_1/3:output:0*
N*
T0*
_output_shapes
:?
SliceSliceinputsstack:output:0stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????@@?????????\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   ?
resize/ResizeBilinearResizeBilinearSlice:output:0resize/size:output:0*
T0*/
_output_shapes
:?????????@@*
half_pixel_centers(v
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2087
xG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@M
:window_attention_dense_1_tensordot_readvariableop_resource:	@?G
8window_attention_dense_1_biasadd_readvariableop_resource:	?D
2window_attention_reshape_1_readvariableop_resource:	2
 window_attention_gather_resource:	L
:window_attention_dense_2_tensordot_readvariableop_resource:@@F
8window_attention_dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@G
4sequential_dense_3_tensordot_readvariableop_resource:	@?A
2sequential_dense_3_biasadd_readvariableop_resource:	?G
4sequential_dense_4_tensordot_readvariableop_resource:	?@@
2sequential_dense_4_biasadd_readvariableop_resource:@
identity??,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?)sequential/dense_3/BiasAdd/ReadVariableOp?+sequential/dense_3/Tensordot/ReadVariableOp?)sequential/dense_4/BiasAdd/ReadVariableOp?+sequential/dense_4/Tensordot/ReadVariableOp?window_attention/Gather?)window_attention/Reshape_1/ReadVariableOp?/window_attention/dense_1/BiasAdd/ReadVariableOp?1window_attention/dense_1/Tensordot/ReadVariableOp?/window_attention/dense_2/BiasAdd/ReadVariableOp?1window_attention/dense_2/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanx;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:???????????
-layer_normalization/moments/SquaredDifferenceSquaredDifferencex1layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/mul_1Mulx%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   ?
ReshapeReshape'layer_normalization/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:?????????  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_1ReshapeReshape:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!?????????@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention/dense_1/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0q
'window_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'window_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       j
(window_attention/dense_1/Tensordot/ShapeShapeReshape_3:output:0*
T0*
_output_shapes
:r
0window_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_1/Tensordot/GatherV2GatherV21window_attention/dense_1/Tensordot/Shape:output:00window_attention/dense_1/Tensordot/free:output:09window_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2window_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention/dense_1/Tensordot/GatherV2_1GatherV21window_attention/dense_1/Tensordot/Shape:output:00window_attention/dense_1/Tensordot/axes:output:0;window_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(window_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
'window_attention/dense_1/Tensordot/ProdProd4window_attention/dense_1/Tensordot/GatherV2:output:01window_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention/dense_1/Tensordot/Prod_1Prod6window_attention/dense_1/Tensordot/GatherV2_1:output:03window_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)window_attention/dense_1/Tensordot/concatConcatV20window_attention/dense_1/Tensordot/free:output:00window_attention/dense_1/Tensordot/axes:output:07window_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(window_attention/dense_1/Tensordot/stackPack0window_attention/dense_1/Tensordot/Prod:output:02window_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
,window_attention/dense_1/Tensordot/transpose	TransposeReshape_3:output:02window_attention/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
*window_attention/dense_1/Tensordot/ReshapeReshape0window_attention/dense_1/Tensordot/transpose:y:01window_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
)window_attention/dense_1/Tensordot/MatMulMatMul3window_attention/dense_1/Tensordot/Reshape:output:09window_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????u
*window_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?r
0window_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_1/Tensordot/concat_1ConcatV24window_attention/dense_1/Tensordot/GatherV2:output:03window_attention/dense_1/Tensordot/Const_2:output:09window_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
"window_attention/dense_1/TensordotReshape3window_attention/dense_1/Tensordot/MatMul:product:04window_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:???????????
/window_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
 window_attention/dense_1/BiasAddBiasAdd+window_attention/dense_1/Tensordot:output:07window_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????{
window_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention/ReshapeReshape)window_attention/dense_1/BiasAdd:output:0'window_attention/Reshape/shape:output:0*
T0*3
_output_shapes!
:?????????|
window_attention/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
window_attention/transpose	Transpose!window_attention/Reshape:output:0(window_attention/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????n
$window_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&window_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&window_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
window_attention/strided_sliceStridedSlicewindow_attention/transpose:y:0-window_attention/strided_slice/stack:output:0/window_attention/strided_slice/stack_1:output:0/window_attention/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskp
&window_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention/strided_slice_1StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_1/stack:output:01window_attention/strided_slice_1/stack_1:output:01window_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskp
&window_attention/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention/strided_slice_2StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_2/stack:output:01window_attention/strided_slice_2/stack_1:output:01window_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_mask[
window_attention/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
window_attention/mulMul'window_attention/strided_slice:output:0window_attention/mul/y:output:0*
T0*/
_output_shapes
:?????????z
!window_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention/transpose_1	Transpose)window_attention/strided_slice_1:output:0*window_attention/transpose_1/perm:output:0*
T0*/
_output_shapes
:??????????
window_attention/matmulBatchMatMulV2window_attention/mul:z:0 window_attention/transpose_1:y:0*
T0*/
_output_shapes
:??????????
)window_attention/Reshape_1/ReadVariableOpReadVariableOp2window_attention_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	s
 window_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
window_attention/Reshape_1Reshape1window_attention/Reshape_1/ReadVariableOp:value:0)window_attention/Reshape_1/shape:output:0*
T0	*
_output_shapes
:?
window_attention/GatherResourceGather window_attention_gather_resource#window_attention/Reshape_1:output:0*
Tindices0	*
_output_shapes

:*
dtype0p
window_attention/IdentityIdentity window_attention/Gather:output:0*
T0*
_output_shapes

:u
 window_attention/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?????
window_attention/Reshape_2Reshape"window_attention/Identity:output:0)window_attention/Reshape_2/shape:output:0*
T0*"
_output_shapes
:v
!window_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
window_attention/transpose_2	Transpose#window_attention/Reshape_2:output:0*window_attention/transpose_2/perm:output:0*
T0*"
_output_shapes
:a
window_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention/ExpandDims
ExpandDims window_attention/transpose_2:y:0(window_attention/ExpandDims/dim:output:0*
T0*&
_output_shapes
:?
window_attention/addAddV2 window_attention/matmul:output:0$window_attention/ExpandDims:output:0*
T0*/
_output_shapes
:?????????w
window_attention/SoftmaxSoftmaxwindow_attention/add:z:0*
T0*/
_output_shapes
:?????????k
&window_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
$window_attention/dropout/dropout/MulMul"window_attention/Softmax:softmax:0/window_attention/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????x
&window_attention/dropout/dropout/ShapeShape"window_attention/Softmax:softmax:0*
T0*
_output_shapes
:?
=window_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform/window_attention/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0t
/window_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
-window_attention/dropout/dropout/GreaterEqualGreaterEqualFwindow_attention/dropout/dropout/random_uniform/RandomUniform:output:08window_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
%window_attention/dropout/dropout/CastCast1window_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
&window_attention/dropout/dropout/Mul_1Mul(window_attention/dropout/dropout/Mul:z:0)window_attention/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
window_attention/matmul_1BatchMatMulV2*window_attention/dropout/dropout/Mul_1:z:0)window_attention/strided_slice_2:output:0*
T0*/
_output_shapes
:?????????z
!window_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention/transpose_3	Transpose"window_attention/matmul_1:output:0*window_attention/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????u
 window_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   ?
window_attention/Reshape_3Reshape window_attention/transpose_3:y:0)window_attention/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention/dense_2/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0q
'window_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'window_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
(window_attention/dense_2/Tensordot/ShapeShape#window_attention/Reshape_3:output:0*
T0*
_output_shapes
:r
0window_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_2/Tensordot/GatherV2GatherV21window_attention/dense_2/Tensordot/Shape:output:00window_attention/dense_2/Tensordot/free:output:09window_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2window_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention/dense_2/Tensordot/GatherV2_1GatherV21window_attention/dense_2/Tensordot/Shape:output:00window_attention/dense_2/Tensordot/axes:output:0;window_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(window_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
'window_attention/dense_2/Tensordot/ProdProd4window_attention/dense_2/Tensordot/GatherV2:output:01window_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention/dense_2/Tensordot/Prod_1Prod6window_attention/dense_2/Tensordot/GatherV2_1:output:03window_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)window_attention/dense_2/Tensordot/concatConcatV20window_attention/dense_2/Tensordot/free:output:00window_attention/dense_2/Tensordot/axes:output:07window_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(window_attention/dense_2/Tensordot/stackPack0window_attention/dense_2/Tensordot/Prod:output:02window_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
,window_attention/dense_2/Tensordot/transpose	Transpose#window_attention/Reshape_3:output:02window_attention/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
*window_attention/dense_2/Tensordot/ReshapeReshape0window_attention/dense_2/Tensordot/transpose:y:01window_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
)window_attention/dense_2/Tensordot/MatMulMatMul3window_attention/dense_2/Tensordot/Reshape:output:09window_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@t
*window_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@r
0window_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_2/Tensordot/concat_1ConcatV24window_attention/dense_2/Tensordot/GatherV2:output:03window_attention/dense_2/Tensordot/Const_2:output:09window_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
"window_attention/dense_2/TensordotReshape3window_attention/dense_2/Tensordot/MatMul:product:04window_attention/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@?
/window_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
 window_attention/dense_2/BiasAddBiasAdd+window_attention/dense_2/Tensordot:output:07window_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@m
(window_attention/dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
&window_attention/dropout/dropout_1/MulMul)window_attention/dense_2/BiasAdd:output:01window_attention/dropout/dropout_1/Const:output:0*
T0*+
_output_shapes
:?????????@?
(window_attention/dropout/dropout_1/ShapeShape)window_attention/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:?
?window_attention/dropout/dropout_1/random_uniform/RandomUniformRandomUniform1window_attention/dropout/dropout_1/Shape:output:0*
T0*+
_output_shapes
:?????????@*
dtype0v
1window_attention/dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
/window_attention/dropout/dropout_1/GreaterEqualGreaterEqualHwindow_attention/dropout/dropout_1/random_uniform/RandomUniform:output:0:window_attention/dropout/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????@?
'window_attention/dropout/dropout_1/CastCast3window_attention/dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????@?
(window_attention/dropout/dropout_1/Mul_1Mul*window_attention/dropout/dropout_1/Mul:z:0+window_attention/dropout/dropout_1/Cast:y:0*
T0*+
_output_shapes
:?????????@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   ?
	Reshape_4Reshape,window_attention/dropout/dropout_1/Mul_1:z:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:?????????@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!?????????@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:?????????  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   y
	Reshape_7ReshapeReshape_6:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:??????????@Q
drop_path/ShapeShapeReshape_7:output:0*
T0*
_output_shapes
:g
drop_path/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
drop_path/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
drop_path/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path/strided_sliceStridedSlicedrop_path/Shape:output:0&drop_path/strided_slice/stack:output:0(drop_path/strided_slice/stack_1:output:0(drop_path/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 drop_path/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :b
 drop_path/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
drop_path/random_uniform/shapePack drop_path/strided_slice:output:0)drop_path/random_uniform/shape/1:output:0)drop_path/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:?
&drop_path/random_uniform/RandomUniformRandomUniform'drop_path/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0T
drop_path/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/addAddV2drop_path/add/x:output:0/drop_path/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:?????????a
drop_path/FloorFloordrop_path/add:z:0*
T0*+
_output_shapes
:?????????X
drop_path/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/truedivRealDivReshape_7:output:0drop_path/truediv/y:output:0*
T0*,
_output_shapes
:??????????@w
drop_path/mulMuldrop_path/truediv:z:0drop_path/Floor:y:0*
T0*,
_output_shapes
:??????????@Y
addAddV2xdrop_path/mul:z:0*
T0*,
_output_shapes
:??????????@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@?
+sequential/dense_3/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0k
!sequential/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
"sequential/dense_3/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_3/Tensordot/GatherV2GatherV2+sequential/dense_3/Tensordot/Shape:output:0*sequential/dense_3/Tensordot/free:output:03sequential/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential/dense_3/Tensordot/GatherV2_1GatherV2+sequential/dense_3/Tensordot/Shape:output:0*sequential/dense_3/Tensordot/axes:output:05sequential/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!sequential/dense_3/Tensordot/ProdProd.sequential/dense_3/Tensordot/GatherV2:output:0+sequential/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
#sequential/dense_3/Tensordot/Prod_1Prod0sequential/dense_3/Tensordot/GatherV2_1:output:0-sequential/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/dense_3/Tensordot/concatConcatV2*sequential/dense_3/Tensordot/free:output:0*sequential/dense_3/Tensordot/axes:output:01sequential/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"sequential/dense_3/Tensordot/stackPack*sequential/dense_3/Tensordot/Prod:output:0,sequential/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
&sequential/dense_3/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0,sequential/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
$sequential/dense_3/Tensordot/ReshapeReshape*sequential/dense_3/Tensordot/transpose:y:0+sequential/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
#sequential/dense_3/Tensordot/MatMulMatMul-sequential/dense_3/Tensordot/Reshape:output:03sequential/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
$sequential/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?l
*sequential/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_3/Tensordot/concat_1ConcatV2.sequential/dense_3/Tensordot/GatherV2:output:0-sequential/dense_3/Tensordot/Const_2:output:03sequential/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential/dense_3/TensordotReshape-sequential/dense_3/Tensordot/MatMul:product:0.sequential/dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/dense_3/BiasAddBiasAdd%sequential/dense_3/Tensordot:output:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????e
 sequential/activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
sequential/activation/Gelu/mulMul)sequential/activation/Gelu/mul/x:output:0#sequential/dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:???????????f
!sequential/activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
"sequential/activation/Gelu/truedivRealDiv#sequential/dense_3/BiasAdd:output:0*sequential/activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:????????????
sequential/activation/Gelu/ErfErf&sequential/activation/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????e
 sequential/activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sequential/activation/Gelu/addAddV2)sequential/activation/Gelu/add/x:output:0"sequential/activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
 sequential/activation/Gelu/mul_1Mul"sequential/activation/Gelu/mul:z:0"sequential/activation/Gelu/add:z:0*
T0*-
_output_shapes
:???????????g
"sequential/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
 sequential/dropout_1/dropout/MulMul$sequential/activation/Gelu/mul_1:z:0+sequential/dropout_1/dropout/Const:output:0*
T0*-
_output_shapes
:???????????v
"sequential/dropout_1/dropout/ShapeShape$sequential/activation/Gelu/mul_1:z:0*
T0*
_output_shapes
:?
9sequential/dropout_1/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_1/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0p
+sequential/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
)sequential/dropout_1/dropout/GreaterEqualGreaterEqualBsequential/dropout_1/dropout/random_uniform/RandomUniform:output:04sequential/dropout_1/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
!sequential/dropout_1/dropout/CastCast-sequential/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
"sequential/dropout_1/dropout/Mul_1Mul$sequential/dropout_1/dropout/Mul:z:0%sequential/dropout_1/dropout/Cast:y:0*
T0*-
_output_shapes
:????????????
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0k
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       x
"sequential/dense_4/Tensordot/ShapeShape&sequential/dropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
&sequential/dense_4/Tensordot/transpose	Transpose&sequential/dropout_1/dropout/Mul_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@g
"sequential/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
 sequential/dropout_2/dropout/MulMul#sequential/dense_4/BiasAdd:output:0+sequential/dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:??????????@u
"sequential/dropout_2/dropout/ShapeShape#sequential/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:?
9sequential/dropout_2/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype0p
+sequential/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
)sequential/dropout_2/dropout/GreaterEqualGreaterEqualBsequential/dropout_2/dropout/random_uniform/RandomUniform:output:04sequential/dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@?
!sequential/dropout_2/dropout/CastCast-sequential/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@?
"sequential/dropout_2/dropout/Mul_1Mul$sequential/dropout_2/dropout/Mul:z:0%sequential/dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@g
drop_path/Shape_1Shape&sequential/dropout_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:i
drop_path/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!drop_path/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!drop_path/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path/strided_slice_1StridedSlicedrop_path/Shape_1:output:0(drop_path/strided_slice_1/stack:output:0*drop_path/strided_slice_1/stack_1:output:0*drop_path/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"drop_path/random_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d
"drop_path/random_uniform_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
 drop_path/random_uniform_1/shapePack"drop_path/strided_slice_1:output:0+drop_path/random_uniform_1/shape/1:output:0+drop_path/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
(drop_path/random_uniform_1/RandomUniformRandomUniform)drop_path/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0V
drop_path/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/add_1AddV2drop_path/add_1/x:output:01drop_path/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:?????????e
drop_path/Floor_1Floordrop_path/add_1:z:0*
T0*+
_output_shapes
:?????????Z
drop_path/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/truediv_1RealDiv&sequential/dropout_2/dropout/Mul_1:z:0drop_path/truediv_1/y:output:0*
T0*,
_output_shapes
:??????????@}
drop_path/mul_1Muldrop_path/truediv_1:z:0drop_path/Floor_1:y:0*
T0*,
_output_shapes
:??????????@c
add_1AddV2add:z:0drop_path/mul_1:z:0*
T0*,
_output_shapes
:??????????@?
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp,^sequential/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp^window_attention/Gather*^window_attention/Reshape_1/ReadVariableOp0^window_attention/dense_1/BiasAdd/ReadVariableOp2^window_attention/dense_1/Tensordot/ReadVariableOp0^window_attention/dense_2/BiasAdd/ReadVariableOp2^window_attention/dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????@: : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2Z
+sequential/dense_3/Tensordot/ReadVariableOp+sequential/dense_3/Tensordot/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp22
window_attention/Gatherwindow_attention/Gather2V
)window_attention/Reshape_1/ReadVariableOp)window_attention/Reshape_1/ReadVariableOp2b
/window_attention/dense_1/BiasAdd/ReadVariableOp/window_attention/dense_1/BiasAdd/ReadVariableOp2f
1window_attention/dense_1/Tensordot/ReadVariableOp1window_attention/dense_1/Tensordot/ReadVariableOp2b
/window_attention/dense_2/BiasAdd/ReadVariableOp/window_attention/dense_2/BiasAdd/ReadVariableOp2f
1window_attention/dense_2/Tensordot/ReadVariableOp1window_attention/dense_2/Tensordot/ReadVariableOp:O K
,
_output_shapes
:??????????@

_user_specified_namex
?i
?
@__inference_model_layer_call_and_return_conditional_losses_11769

inputs&
random_crop_cond_input_1:	+
random_flip_map_while_input_6:	'
patch_embedding_11689:@#
patch_embedding_11691:@(
patch_embedding_11693:	?@$
swin_transformer_11696:@$
swin_transformer_11698:@)
swin_transformer_11700:	@?%
swin_transformer_11702:	?(
swin_transformer_11704:	(
swin_transformer_11706:	(
swin_transformer_11708:@@$
swin_transformer_11710:@$
swin_transformer_11712:@$
swin_transformer_11714:@)
swin_transformer_11716:	@?%
swin_transformer_11718:	?)
swin_transformer_11720:	?@$
swin_transformer_11722:@&
swin_transformer_1_11725:@&
swin_transformer_1_11727:@+
swin_transformer_1_11729:	@?'
swin_transformer_1_11731:	?*
swin_transformer_1_11733:	*
swin_transformer_1_11735:	/
swin_transformer_1_11737:?*
swin_transformer_1_11739:@@&
swin_transformer_1_11741:@&
swin_transformer_1_11743:@&
swin_transformer_1_11745:@+
swin_transformer_1_11747:	@?'
swin_transformer_1_11749:	?+
swin_transformer_1_11751:	?@&
swin_transformer_1_11753:@'
patch_merging_11756:
??:
'dense_10_matmul_readvariableop_resource:	?W6
(dense_10_biasadd_readvariableop_resource:W
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?'patch_embedding/StatefulPartitionedCall?%patch_merging/StatefulPartitionedCall?random_crop/cond?random_flip/map/while?(swin_transformer/StatefulPartitionedCall?*swin_transformer_1/StatefulPartitionedCallG
random_crop/ShapeShapeinputs*
T0*
_output_shapes
:r
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????t
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????k
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/strided_sliceStridedSlicerandom_crop/Shape:output:0(random_crop/strided_slice/stack:output:0*random_crop/strided_slice/stack_1:output:0*random_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskS
random_crop/sub/yConst*
_output_shapes
: *
dtype0*
value	B :@w
random_crop/subSub"random_crop/strided_slice:output:0random_crop/sub/y:output:0*
T0*
_output_shapes
: t
!random_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop/strided_slice_1StridedSlicerandom_crop/Shape:output:0*random_crop/strided_slice_1/stack:output:0,random_crop/strided_slice_1/stack_1:output:0,random_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
random_crop/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :@}
random_crop/sub_1Sub$random_crop/strided_slice_1:output:0random_crop/sub_1/y:output:0*
T0*
_output_shapes
: \
random_crop/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/GreaterEqualGreaterEqualrandom_crop/sub:z:0#random_crop/GreaterEqual/y:output:0*
T0*
_output_shapes
: ^
random_crop/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop/GreaterEqual_1GreaterEqualrandom_crop/sub_1:z:0%random_crop/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
random_crop/Rank/packedPackrandom_crop/GreaterEqual:z:0random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:R
random_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
random_crop/range/startConst*
_output_shapes
: *
dtype0*
value	B : Y
random_crop/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop/rangeRange random_crop/range/start:output:0random_crop/Rank:output:0 random_crop/range/delta:output:0*
_output_shapes
:?
random_crop/All/inputPackrandom_crop/GreaterEqual:z:0random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:j
random_crop/AllAllrandom_crop/All/input:output:0random_crop/range:output:0*
_output_shapes
: ?
random_crop/condIfrandom_crop/All:output:0inputsrandom_crop_cond_input_1*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 */
else_branch R
random_crop_cond_false_11439*.
output_shapes
:?????????@@*.
then_branchR
random_crop_cond_true_11438z
random_crop/cond/IdentityIdentityrandom_crop/cond:output:0*
T0*/
_output_shapes
:?????????@@g
random_flip/map/ShapeShape"random_crop/cond/Identity:output:0*
T0*
_output_shapes
:m
#random_flip/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%random_flip/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%random_flip/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_flip/map/strided_sliceStridedSlicerandom_flip/map/Shape:output:0,random_flip/map/strided_slice/stack:output:0.random_flip/map/strided_slice/stack_1:output:0.random_flip/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+random_flip/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
random_flip/map/TensorArrayV2TensorListReserve4random_flip/map/TensorArrayV2/element_shape:output:0&random_flip/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Erandom_flip/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      ?
7random_flip/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"random_crop/cond/Identity:output:0Nrandom_flip/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???W
random_flip/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : x
-random_flip/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
random_flip/map/TensorArrayV2_1TensorListReserve6random_flip/map/TensorArrayV2_1/element_shape:output:0&random_flip/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???d
"random_flip/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
random_flip/map/whileWhile+random_flip/map/while/loop_counter:output:0&random_flip/map/strided_slice:output:0random_flip/map/Const:output:0(random_flip/map/TensorArrayV2_1:handle:0&random_flip/map/strided_slice:output:0Grandom_flip/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0random_flip_map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *,
body$R"
 random_flip_map_while_body_11596*,
cond$R"
 random_flip_map_while_cond_11595*!
output_shapes
: : : : : : : ?
@random_flip/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      ?
2random_flip/map/TensorArrayV2Stack/TensorListStackTensorListStackrandom_flip/map/while:output:3Irandom_flip/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*/
_output_shapes
:?????????@@*
element_dtype0?
patch_extract/PartitionedCallPartitionedCall;random_flip/map/TensorArrayV2Stack/TensorListStack:tensor:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9773?
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_11689patch_embedding_11691patch_embedding_11693*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9785?
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_11696swin_transformer_11698swin_transformer_11700swin_transformer_11702swin_transformer_11704swin_transformer_11706swin_transformer_11708swin_transformer_11710swin_transformer_11712swin_transformer_11714swin_transformer_11716swin_transformer_11718swin_transformer_11720swin_transformer_11722*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *1
f,R*
(__inference_restored_function_body_10622?
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_11725swin_transformer_1_11727swin_transformer_1_11729swin_transformer_1_11731swin_transformer_1_11733swin_transformer_1_11735swin_transformer_1_11737swin_transformer_1_11739swin_transformer_1_11741swin_transformer_1_11743swin_transformer_1_11745swin_transformer_1_11747swin_transformer_1_11749swin_transformer_1_11751swin_transformer_1_11753*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *1
f,R*
(__inference_restored_function_body_10686?
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_11756*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restored_function_body_9927q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d/MeanMean.patch_merging/StatefulPartitionedCall:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	?W*
dtype0?
dense_10/MatMulMatMul&global_average_pooling1d/Mean:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????W?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:W*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Wh
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Wi
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????W?
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall^random_crop/cond^random_flip/map/while)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2$
random_crop/condrandom_crop/cond2.
random_flip/map/whilerandom_flip/map/while2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_12264

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q???i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
map_while_cond_12031$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice;
7map_while_map_while_cond_12031___redundant_placeholder0;
7map_while_map_while_cond_12031___redundant_placeholder1
map_while_identity
p
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: x
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
?
?
'__inference_restored_function_body_9785	
patch
unknown:@
	unknown_0:@
	unknown_1:	?@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallpatchunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*,
_output_shapes
:??????????@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_patch_embedding_layer_call_and_return_conditional_losses_2689t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:??????????

_user_specified_namepatch
?
?
F__inference_random_flip_layer_call_and_return_conditional_losses_10305

inputs
map_while_input_6:	
identity??	map/while?
	map/ShapeShapeinputs*
T0*
_output_shapes
:a
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      ?
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputsBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???K
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???X
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( * 
bodyR
map_while_body_10212* 
condR
map_while_cond_10211*!
output_shapes
: : : : : : : ?
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      ?
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*/
_output_shapes
:?????????@@*
element_dtype0?
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0^NoOp*
T0*/
_output_shapes
:?????????@@R
NoOpNoOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: 2
	map/while	map/while:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
b
F__inference_random_flip_layer_call_and_return_conditional_losses_10013

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12514

inputs 
dense_7_12444:	@?
dense_7_12446:	? 
dense_8_12501:	?@
dense_8_12503:@
identity??dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_12444dense_7_12446*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_12443?
activation_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_12461?
dropout_4/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12468?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_8_12501dense_8_12503*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_12500?
dropout_5/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12511v
IdentityIdentity"dropout_5/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12629

inputs 
dense_7_12615:	@?
dense_7_12617:	? 
dense_8_12622:	?@
dense_8_12624:@
identity??dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_12615dense_7_12617*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_12443?
activation_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_12461?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12578?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_8_12622dense_8_12624*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_12500?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12545~
IdentityIdentity*dropout_5/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_13133

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12187f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
map_while_cond_10211$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice;
7map_while_map_while_cond_10211___redundant_placeholder0;
7map_while_map_while_cond_10211___redundant_placeholder1
map_while_identity
p
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: x
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
?
b
)__inference_dropout_1_layer_call_fn_13138

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12297u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_5_layer_call_fn_13348

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12511e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
(__inference_dense_10_layer_call_fn_12707

inputs
unknown:	?W
	unknown_0:W
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_10098o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Brandom_flip_map_while_stateless_random_flip_left_right_false_11656?
?random_flip_map_while_stateless_random_flip_left_right_identity_random_flip_map_while_stateless_random_flip_left_right_control_dependencyC
?random_flip_map_while_stateless_random_flip_left_right_identity?
?random_flip/map/while/stateless_random_flip_left_right/IdentityIdentity?random_flip_map_while_stateless_random_flip_left_right_identity_random_flip_map_while_stateless_random_flip_left_right_control_dependency*
T0*"
_output_shapes
:@@"?
?random_flip_map_while_stateless_random_flip_left_right_identityHrandom_flip/map/while/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:@@:( $
"
_output_shapes
:@@
?
F
*__inference_activation_layer_call_fn_13116

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12180f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
I__inference_swin_transformer_layer_call_and_return_conditional_losses_721
xG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@M
:window_attention_dense_1_tensordot_readvariableop_resource:	@?G
8window_attention_dense_1_biasadd_readvariableop_resource:	?D
2window_attention_reshape_1_readvariableop_resource:	2
 window_attention_gather_resource:	L
:window_attention_dense_2_tensordot_readvariableop_resource:@@F
8window_attention_dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@G
4sequential_dense_3_tensordot_readvariableop_resource:	@?A
2sequential_dense_3_biasadd_readvariableop_resource:	?G
4sequential_dense_4_tensordot_readvariableop_resource:	?@@
2sequential_dense_4_biasadd_readvariableop_resource:@
identity??,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?)sequential/dense_3/BiasAdd/ReadVariableOp?+sequential/dense_3/Tensordot/ReadVariableOp?)sequential/dense_4/BiasAdd/ReadVariableOp?+sequential/dense_4/Tensordot/ReadVariableOp?window_attention/Gather?)window_attention/Reshape_1/ReadVariableOp?/window_attention/dense_1/BiasAdd/ReadVariableOp?1window_attention/dense_1/Tensordot/ReadVariableOp?/window_attention/dense_2/BiasAdd/ReadVariableOp?1window_attention/dense_2/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanx;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:???????????
-layer_normalization/moments/SquaredDifferenceSquaredDifferencex1layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/mul_1Mulx%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   ?
ReshapeReshape'layer_normalization/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:?????????  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_1ReshapeReshape:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!?????????@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention/dense_1/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0q
'window_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'window_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       j
(window_attention/dense_1/Tensordot/ShapeShapeReshape_3:output:0*
T0*
_output_shapes
:r
0window_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_1/Tensordot/GatherV2GatherV21window_attention/dense_1/Tensordot/Shape:output:00window_attention/dense_1/Tensordot/free:output:09window_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2window_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention/dense_1/Tensordot/GatherV2_1GatherV21window_attention/dense_1/Tensordot/Shape:output:00window_attention/dense_1/Tensordot/axes:output:0;window_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(window_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
'window_attention/dense_1/Tensordot/ProdProd4window_attention/dense_1/Tensordot/GatherV2:output:01window_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention/dense_1/Tensordot/Prod_1Prod6window_attention/dense_1/Tensordot/GatherV2_1:output:03window_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)window_attention/dense_1/Tensordot/concatConcatV20window_attention/dense_1/Tensordot/free:output:00window_attention/dense_1/Tensordot/axes:output:07window_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(window_attention/dense_1/Tensordot/stackPack0window_attention/dense_1/Tensordot/Prod:output:02window_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
,window_attention/dense_1/Tensordot/transpose	TransposeReshape_3:output:02window_attention/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
*window_attention/dense_1/Tensordot/ReshapeReshape0window_attention/dense_1/Tensordot/transpose:y:01window_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
)window_attention/dense_1/Tensordot/MatMulMatMul3window_attention/dense_1/Tensordot/Reshape:output:09window_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????u
*window_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?r
0window_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_1/Tensordot/concat_1ConcatV24window_attention/dense_1/Tensordot/GatherV2:output:03window_attention/dense_1/Tensordot/Const_2:output:09window_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
"window_attention/dense_1/TensordotReshape3window_attention/dense_1/Tensordot/MatMul:product:04window_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:???????????
/window_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
 window_attention/dense_1/BiasAddBiasAdd+window_attention/dense_1/Tensordot:output:07window_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????{
window_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"????            ?
window_attention/ReshapeReshape)window_attention/dense_1/BiasAdd:output:0'window_attention/Reshape/shape:output:0*
T0*3
_output_shapes!
:?????????|
window_attention/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
window_attention/transpose	Transpose!window_attention/Reshape:output:0(window_attention/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????n
$window_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&window_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&window_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
window_attention/strided_sliceStridedSlicewindow_attention/transpose:y:0-window_attention/strided_slice/stack:output:0/window_attention/strided_slice/stack_1:output:0/window_attention/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskp
&window_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention/strided_slice_1StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_1/stack:output:01window_attention/strided_slice_1/stack_1:output:01window_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_maskp
&window_attention/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(window_attention/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 window_attention/strided_slice_2StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_2/stack:output:01window_attention/strided_slice_2/stack_1:output:01window_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????*
shrink_axis_mask[
window_attention/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
window_attention/mulMul'window_attention/strided_slice:output:0window_attention/mul/y:output:0*
T0*/
_output_shapes
:?????????z
!window_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention/transpose_1	Transpose)window_attention/strided_slice_1:output:0*window_attention/transpose_1/perm:output:0*
T0*/
_output_shapes
:??????????
window_attention/matmulBatchMatMulV2window_attention/mul:z:0 window_attention/transpose_1:y:0*
T0*/
_output_shapes
:??????????
)window_attention/Reshape_1/ReadVariableOpReadVariableOp2window_attention_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	s
 window_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
window_attention/Reshape_1Reshape1window_attention/Reshape_1/ReadVariableOp:value:0)window_attention/Reshape_1/shape:output:0*
T0	*
_output_shapes
:?
window_attention/GatherResourceGather window_attention_gather_resource#window_attention/Reshape_1:output:0*
Tindices0	*
_output_shapes

:*
dtype0p
window_attention/IdentityIdentity window_attention/Gather:output:0*
T0*
_output_shapes

:u
 window_attention/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ?????
window_attention/Reshape_2Reshape"window_attention/Identity:output:0)window_attention/Reshape_2/shape:output:0*
T0*"
_output_shapes
:v
!window_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
window_attention/transpose_2	Transpose#window_attention/Reshape_2:output:0*window_attention/transpose_2/perm:output:0*
T0*"
_output_shapes
:a
window_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
window_attention/ExpandDims
ExpandDims window_attention/transpose_2:y:0(window_attention/ExpandDims/dim:output:0*
T0*&
_output_shapes
:?
window_attention/addAddV2 window_attention/matmul:output:0$window_attention/ExpandDims:output:0*
T0*/
_output_shapes
:?????????w
window_attention/SoftmaxSoftmaxwindow_attention/add:z:0*
T0*/
_output_shapes
:??????????
!window_attention/dropout/IdentityIdentity"window_attention/Softmax:softmax:0*
T0*/
_output_shapes
:??????????
window_attention/matmul_1BatchMatMulV2*window_attention/dropout/Identity:output:0)window_attention/strided_slice_2:output:0*
T0*/
_output_shapes
:?????????z
!window_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ?
window_attention/transpose_3	Transpose"window_attention/matmul_1:output:0*window_attention/transpose_3/perm:output:0*
T0*/
_output_shapes
:?????????u
 window_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   ?
window_attention/Reshape_3Reshape window_attention/transpose_3:y:0)window_attention/Reshape_3/shape:output:0*
T0*+
_output_shapes
:?????????@?
1window_attention/dense_2/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_2_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0q
'window_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'window_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
(window_attention/dense_2/Tensordot/ShapeShape#window_attention/Reshape_3:output:0*
T0*
_output_shapes
:r
0window_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_2/Tensordot/GatherV2GatherV21window_attention/dense_2/Tensordot/Shape:output:00window_attention/dense_2/Tensordot/free:output:09window_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2window_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-window_attention/dense_2/Tensordot/GatherV2_1GatherV21window_attention/dense_2/Tensordot/Shape:output:00window_attention/dense_2/Tensordot/axes:output:0;window_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(window_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
'window_attention/dense_2/Tensordot/ProdProd4window_attention/dense_2/Tensordot/GatherV2:output:01window_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
)window_attention/dense_2/Tensordot/Prod_1Prod6window_attention/dense_2/Tensordot/GatherV2_1:output:03window_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)window_attention/dense_2/Tensordot/concatConcatV20window_attention/dense_2/Tensordot/free:output:00window_attention/dense_2/Tensordot/axes:output:07window_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
(window_attention/dense_2/Tensordot/stackPack0window_attention/dense_2/Tensordot/Prod:output:02window_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
,window_attention/dense_2/Tensordot/transpose	Transpose#window_attention/Reshape_3:output:02window_attention/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@?
*window_attention/dense_2/Tensordot/ReshapeReshape0window_attention/dense_2/Tensordot/transpose:y:01window_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
)window_attention/dense_2/Tensordot/MatMulMatMul3window_attention/dense_2/Tensordot/Reshape:output:09window_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@t
*window_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@r
0window_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+window_attention/dense_2/Tensordot/concat_1ConcatV24window_attention/dense_2/Tensordot/GatherV2:output:03window_attention/dense_2/Tensordot/Const_2:output:09window_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
"window_attention/dense_2/TensordotReshape3window_attention/dense_2/Tensordot/MatMul:product:04window_attention/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????@?
/window_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
 window_attention/dense_2/BiasAddBiasAdd+window_attention/dense_2/Tensordot:output:07window_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@?
#window_attention/dropout/Identity_1Identity)window_attention/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????      @   ?
	Reshape_4Reshape,window_attention/dropout/Identity_1:output:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:?????????@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""????            @   ?
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!?????????@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   ?
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!?????????@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:?????????  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????   @   y
	Reshape_7ReshapeReshape_6:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:??????????@Q
drop_path/ShapeShapeReshape_7:output:0*
T0*
_output_shapes
:g
drop_path/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
drop_path/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
drop_path/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path/strided_sliceStridedSlicedrop_path/Shape:output:0&drop_path/strided_slice/stack:output:0(drop_path/strided_slice/stack_1:output:0(drop_path/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 drop_path/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :b
 drop_path/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
drop_path/random_uniform/shapePack drop_path/strided_slice:output:0)drop_path/random_uniform/shape/1:output:0)drop_path/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:?
&drop_path/random_uniform/RandomUniformRandomUniform'drop_path/random_uniform/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0T
drop_path/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/addAddV2drop_path/add/x:output:0/drop_path/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:?????????a
drop_path/FloorFloordrop_path/add:z:0*
T0*+
_output_shapes
:?????????X
drop_path/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/truedivRealDivReshape_7:output:0drop_path/truediv/y:output:0*
T0*,
_output_shapes
:??????????@w
drop_path/mulMuldrop_path/truediv:z:0drop_path/Floor:y:0*
T0*,
_output_shapes
:??????????@Y
addAddV2xdrop_path/mul:z:0*
T0*,
_output_shapes
:??????????@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:???????????
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????@?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:???????????
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:???????????
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????@?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????@?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????@?
+sequential/dense_3/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0k
!sequential/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
"sequential/dense_3/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:l
*sequential/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_3/Tensordot/GatherV2GatherV2+sequential/dense_3/Tensordot/Shape:output:0*sequential/dense_3/Tensordot/free:output:03sequential/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential/dense_3/Tensordot/GatherV2_1GatherV2+sequential/dense_3/Tensordot/Shape:output:0*sequential/dense_3/Tensordot/axes:output:05sequential/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!sequential/dense_3/Tensordot/ProdProd.sequential/dense_3/Tensordot/GatherV2:output:0+sequential/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
#sequential/dense_3/Tensordot/Prod_1Prod0sequential/dense_3/Tensordot/GatherV2_1:output:0-sequential/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/dense_3/Tensordot/concatConcatV2*sequential/dense_3/Tensordot/free:output:0*sequential/dense_3/Tensordot/axes:output:01sequential/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"sequential/dense_3/Tensordot/stackPack*sequential/dense_3/Tensordot/Prod:output:0,sequential/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
&sequential/dense_3/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0,sequential/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
$sequential/dense_3/Tensordot/ReshapeReshape*sequential/dense_3/Tensordot/transpose:y:0+sequential/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
#sequential/dense_3/Tensordot/MatMulMatMul-sequential/dense_3/Tensordot/Reshape:output:03sequential/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
$sequential/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?l
*sequential/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_3/Tensordot/concat_1ConcatV2.sequential/dense_3/Tensordot/GatherV2:output:0-sequential/dense_3/Tensordot/Const_2:output:03sequential/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential/dense_3/TensordotReshape-sequential/dense_3/Tensordot/MatMul:product:0.sequential/dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential/dense_3/BiasAddBiasAdd%sequential/dense_3/Tensordot:output:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????e
 sequential/activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
sequential/activation/Gelu/mulMul)sequential/activation/Gelu/mul/x:output:0#sequential/dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:???????????f
!sequential/activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
"sequential/activation/Gelu/truedivRealDiv#sequential/dense_3/BiasAdd:output:0*sequential/activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:????????????
sequential/activation/Gelu/ErfErf&sequential/activation/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????e
 sequential/activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sequential/activation/Gelu/addAddV2)sequential/activation/Gelu/add/x:output:0"sequential/activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
 sequential/activation/Gelu/mul_1Mul"sequential/activation/Gelu/mul:z:0"sequential/activation/Gelu/add:z:0*
T0*-
_output_shapes
:????????????
sequential/dropout_1/IdentityIdentity$sequential/activation/Gelu/mul_1:z:0*
T0*-
_output_shapes
:????????????
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0k
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       x
"sequential/dense_4/Tensordot/ShapeShape&sequential/dropout_1/Identity:output:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
&sequential/dense_4/Tensordot/transpose	Transpose&sequential/dropout_1/Identity:output:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@?
sequential/dropout_2/IdentityIdentity#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@g
drop_path/Shape_1Shape&sequential/dropout_2/Identity:output:0*
T0*
_output_shapes
:i
drop_path/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!drop_path/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!drop_path/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
drop_path/strided_slice_1StridedSlicedrop_path/Shape_1:output:0(drop_path/strided_slice_1/stack:output:0*drop_path/strided_slice_1/stack_1:output:0*drop_path/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"drop_path/random_uniform_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d
"drop_path/random_uniform_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
 drop_path/random_uniform_1/shapePack"drop_path/strided_slice_1:output:0+drop_path/random_uniform_1/shape/1:output:0+drop_path/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:?
(drop_path/random_uniform_1/RandomUniformRandomUniform)drop_path/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:?????????*
dtype0V
drop_path/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/add_1AddV2drop_path/add_1/x:output:01drop_path/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:?????????e
drop_path/Floor_1Floordrop_path/add_1:z:0*
T0*+
_output_shapes
:?????????Z
drop_path/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *?Qx??
drop_path/truediv_1RealDiv&sequential/dropout_2/Identity:output:0drop_path/truediv_1/y:output:0*
T0*,
_output_shapes
:??????????@}
drop_path/mul_1Muldrop_path/truediv_1:z:0drop_path/Floor_1:y:0*
T0*,
_output_shapes
:??????????@c
add_1AddV2add:z:0drop_path/mul_1:z:0*
T0*,
_output_shapes
:??????????@?
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp,^sequential/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp^window_attention/Gather*^window_attention/Reshape_1/ReadVariableOp0^window_attention/dense_1/BiasAdd/ReadVariableOp2^window_attention/dense_1/Tensordot/ReadVariableOp0^window_attention/dense_2/BiasAdd/ReadVariableOp2^window_attention/dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:??????????@: : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2Z
+sequential/dense_3/Tensordot/ReadVariableOp+sequential/dense_3/Tensordot/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp22
window_attention/Gatherwindow_attention/Gather2V
)window_attention/Reshape_1/ReadVariableOp)window_attention/Reshape_1/ReadVariableOp2b
/window_attention/dense_1/BiasAdd/ReadVariableOp/window_attention/dense_1/BiasAdd/ReadVariableOp2f
1window_attention/dense_1/Tensordot/ReadVariableOp1window_attention/dense_1/Tensordot/ReadVariableOp2b
/window_attention/dense_2/BiasAdd/ReadVariableOp/window_attention/dense_2/BiasAdd/ReadVariableOp2f
1window_attention/dense_2/Tensordot/ReadVariableOp1window_attention/dense_2/Tensordot/ReadVariableOp:O K
,
_output_shapes
:??????????@

_user_specified_namex
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_12698

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_dense_8_layer_call_and_return_conditional_losses_12500

inputs4
!tensordot_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:????????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?V
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_13072

inputs<
)dense_7_tensordot_readvariableop_resource:	@?6
'dense_7_biasadd_readvariableop_resource:	?<
)dense_8_tensordot_readvariableop_resource:	?@5
'dense_8_biasadd_readvariableop_resource:@
identity??dense_7/BiasAdd/ReadVariableOp? dense_7/Tensordot/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp? dense_8/Tensordot/ReadVariableOp?
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@?*
dtype0`
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       M
dense_7/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:a
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_7/Tensordot/transpose	Transposeinputs!dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@?
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????\
activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
activation_1/Gelu/mulMul activation_1/Gelu/mul/x:output:0dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:???????????]
activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *????
activation_1/Gelu/truedivRealDivdense_7/BiasAdd:output:0!activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:???????????s
activation_1/Gelu/ErfErfactivation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:???????????\
activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
activation_1/Gelu/addAddV2 activation_1/Gelu/add/x:output:0activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:????????????
activation_1/Gelu/mul_1Mulactivation_1/Gelu/mul:z:0activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:???????????\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
dropout_4/dropout/MulMulactivation_1/Gelu/mul_1:z:0 dropout_4/dropout/Const:output:0*
T0*-
_output_shapes
:???????????b
dropout_4/dropout/ShapeShapeactivation_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*-
_output_shapes
:????????????
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?@*
dtype0`
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_8/Tensordot/ShapeShapedropout_4/dropout/Mul_1:z:0*
T0*
_output_shapes
:a
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_8/Tensordot/transpose	Transposedropout_4/dropout/Mul_1:z:0!dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@c
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q????
dropout_5/dropout/MulMuldense_8/BiasAdd:output:0 dropout_5/dropout/Const:output:0*
T0*,
_output_shapes
:??????????@_
dropout_5/dropout/ShapeShapedense_8/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@o
IdentityIdentitydropout_5/dropout/Mul_1:z:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?&
R
cond_false_10346
cond_shape_inputs
cond_placeholder
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :@f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Bf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"?????????@@?????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:?????????@@*
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:?????????@@"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: :5 1
/
_output_shapes
:?????????@@
?
{
+__inference_random_crop_layer_call_fn_11781

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10491w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_10884
input_1
unknown:	
	unknown_0:	
	unknown_1:@
	unknown_2:@
	unknown_3:	?@
	unknown_4:@
	unknown_5:@
	unknown_6:	@?
	unknown_7:	?
	unknown_8:	
	unknown_9:	

unknown_10:@@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:	@?

unknown_15:	?

unknown_16:	?@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:	@?

unknown_21:	?

unknown_22:	

unknown_23:	!

unknown_24:?

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:	@?

unknown_30:	?

unknown_31:	?@

unknown_32:@

unknown_33:
??

unknown_34:	?W

unknown_35:W
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*E
_read_only_resource_inputs'
%#	
 !"#$%*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10728o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@
!
_user_specified_name	input_1
?

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_13155

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *q???j
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???<?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:???????????u
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:???????????o
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:???????????_
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_12233

inputs 
dense_3_12163:	@?
dense_3_12165:	? 
dense_4_12220:	?@
dense_4_12222:@
identity??dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_12163dense_3_12165*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_12162?
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12180?
dropout_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12187?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_4_12220dense_4_12222*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_12219?
dropout_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12230v
IdentityIdentity"dropout_2/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@?
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????@: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
b
)__inference_dropout_4_layer_call_fn_13287

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12578u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_dense_7_layer_call_fn_13230

inputs
unknown:	@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_12443u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????@@<
dense_100
StatefulPartitionedCall:0?????????Wtensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures
#_self_saveable_object_factories"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
#_self_saveable_object_factories"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_random_generator
#$_self_saveable_object_factories"
_tf_keras_layer
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
#+_self_saveable_object_factories"
_tf_keras_layer
?
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2proj
3	pos_embed
#4_self_saveable_object_factories"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
	;norm1
<attn
=	drop_path
	>norm2
?mlp
#@_self_saveable_object_factories"
_tf_keras_layer
?
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
	Gnorm1
Hattn
I	drop_path
	Jnorm2
Kmlp
L	attn_mask
#M_self_saveable_object_factories"
_tf_keras_layer
?
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Tlinear_trans
#U_self_saveable_object_factories"
_tf_keras_layer
?
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
#\_self_saveable_object_factories"
_tf_keras_layer
?
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias
#e_self_saveable_object_factories"
_tf_keras_layer
?
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11
r12
s13
t14
u15
v16
w17
x18
y19
z20
{21
|22
}23
~24
25
?26
?27
?28
?29
L30
?31
?32
c33
d34"
trackable_list_wrapper
?
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11
r12
s13
t14
u15
w16
x17
y18
z19
{20
|21
}22
~23
24
?25
?26
?27
?28
?29
c30
d31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
%__inference_model_layer_call_fn_10178
%__inference_model_layer_call_fn_11206
%__inference_model_layer_call_fn_11285
%__inference_model_layer_call_fn_10884?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
@__inference_model_layer_call_and_return_conditional_losses_11411
@__inference_model_layer_call_and_return_conditional_losses_11769
@__inference_model_layer_call_and_return_conditional_losses_10967
@__inference_model_layer_call_and_return_conditional_losses_11054?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
__inference__wrapped_model_9941input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_random_crop_layer_call_fn_11774
+__inference_random_crop_layer_call_fn_11781?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_random_crop_layer_call_and_return_conditional_losses_11827
F__inference_random_crop_layer_call_and_return_conditional_losses_12000?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
/
?
_generator"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_random_flip_layer_call_fn_12005
+__inference_random_flip_layer_call_fn_12012?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_random_flip_layer_call_and_return_conditional_losses_12016
F__inference_random_flip_layer_call_and_return_conditional_losses_12125?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
/
?
_generator"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_patch_extract_layer_call_fn_3752?
???
FullArgSpec
args?

jimages
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
G__inference_patch_extract_layer_call_and_return_conditional_losses_1291?
???
FullArgSpec
args?

jimages
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
5
f0
g1
h2"
trackable_list_wrapper
5
f0
g1
h2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
.__inference_patch_embedding_layer_call_fn_1272?
???
FullArgSpec
args?	
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
I__inference_patch_embedding_layer_call_and_return_conditional_losses_2689?
???
FullArgSpec
args?	
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

fkernel
gbias
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
h
embeddings
$?_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
?
i0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12
v13"
trackable_list_wrapper
~
i0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
/__inference_swin_transformer_layer_call_fn_6442
/__inference_swin_transformer_layer_call_fn_2650?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
I__inference_swin_transformer_layer_call_and_return_conditional_losses_721
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2087?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	igamma
jbeta
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?qkv
?dropout
	?proj

kweight
 krelative_position_bias_table
vrelative_position_index
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	pgamma
qbeta
$?_self_saveable_object_factories"
_tf_keras_layer
?
?layer_with_weights-0
?layer-0
?layer-1
?layer-2
?layer_with_weights-1
?layer-3
?layer-4
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_sequential
 "
trackable_dict_wrapper
?
w0
x1
y2
z3
{4
|5
}6
~7
8
?9
?10
?11
?12
L13
?14"
trackable_list_wrapper
?
w0
x1
y2
z3
{4
|5
}6
~7
8
?9
?10
?11
?12"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
1__inference_swin_transformer_1_layer_call_fn_1176
1__inference_swin_transformer_1_layer_call_fn_1623?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_2354
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_6078?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	wgamma
xbeta
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?qkv
?dropout
	?proj

yweight
 yrelative_position_bias_table
?relative_position_index
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	~gamma
beta
$?_self_saveable_object_factories"
_tf_keras_layer
?
?layer_with_weights-0
?layer-0
?layer-1
?layer-2
?layer_with_weights-1
?layer-3
?layer-4
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_sequential
0:.?2swin_transformer_1/Variable
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_patch_merging_layer_call_fn_2744?
???
FullArgSpec
args?
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
G__inference_patch_merging_layer_call_and_return_conditional_losses_2938?
???
FullArgSpec
args?
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
$?_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
8__inference_global_average_pooling1d_layer_call_fn_12692?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_12698?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_10_layer_call_fn_12707?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_10_layer_call_and_return_conditional_losses_12718?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
": 	?W2dense_10/kernel
:W2dense_10/bias
 "
trackable_dict_wrapper
.:,@2patch_embedding/dense/kernel
(:&@2patch_embedding/dense/bias
7:5	?@2$patch_embedding/embedding/embeddings
8:6@2*swin_transformer/layer_normalization/gamma
7:5@2)swin_transformer/layer_normalization/beta
::8	2(swin_transformer/window_attention/weight
C:A	@?20swin_transformer/window_attention/dense_1/kernel
=:;?2.swin_transformer/window_attention/dense_1/bias
B:@@@20swin_transformer/window_attention/dense_2/kernel
<::@2.swin_transformer/window_attention/dense_2/bias
::8@2,swin_transformer/layer_normalization_1/gamma
9:7@2+swin_transformer/layer_normalization_1/beta
!:	@?2dense_3/kernel
:?2dense_3/bias
!:	?@2dense_4/kernel
:@2dense_4/bias
::8	2*swin_transformer/window_attention/Variable
<::@2.swin_transformer_1/layer_normalization_2/gamma
;:9@2-swin_transformer_1/layer_normalization_2/beta
>:<	2,swin_transformer_1/window_attention_1/weight
G:E	@?24swin_transformer_1/window_attention_1/dense_5/kernel
A:??22swin_transformer_1/window_attention_1/dense_5/bias
F:D@@24swin_transformer_1/window_attention_1/dense_6/kernel
@:>@22swin_transformer_1/window_attention_1/dense_6/bias
<::@2.swin_transformer_1/layer_normalization_3/gamma
;:9@2-swin_transformer_1/layer_normalization_3/beta
!:	@?2dense_7/kernel
:?2dense_7/bias
!:	?@2dense_8/kernel
:@2dense_8/bias
>:<	2.swin_transformer_1/window_attention_1/Variable
0:.
??2patch_merging/dense_9/kernel
6
v0
L1
?2"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_model_layer_call_fn_10178input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_11206inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_11285inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_10884input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_11411inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_11769inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_10967input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_11054input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_11131input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
+__inference_random_crop_layer_call_fn_11774inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
+__inference_random_crop_layer_call_fn_11781inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_random_crop_layer_call_and_return_conditional_losses_11827inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_random_crop_layer_call_and_return_conditional_losses_12000inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
/
?
_state_var"
_generic_user_object
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
?B?
+__inference_random_flip_layer_call_fn_12005inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
+__inference_random_flip_layer_call_fn_12012inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_random_flip_layer_call_and_return_conditional_losses_12016inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_random_flip_layer_call_and_return_conditional_losses_12125inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
/
?
_state_var"
_generic_user_object
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
?B?
,__inference_patch_extract_layer_call_fn_3752"?
???
FullArgSpec
args?

jimages
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_patch_extract_layer_call_and_return_conditional_losses_1291"?
???
FullArgSpec
args?

jimages
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
.__inference_patch_embedding_layer_call_fn_1272"?
???
FullArgSpec
args?	
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_patch_embedding_layer_call_and_return_conditional_losses_2689"?
???
FullArgSpec
args?	
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
'
h0"
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
'
v0"
trackable_list_wrapper
C
;0
<1
=2
>3
?4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_swin_transformer_layer_call_fn_6442"?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
/__inference_swin_transformer_layer_call_fn_2650"?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_swin_transformer_layer_call_and_return_conditional_losses_721"?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2087"?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
k0
l1
m2
n3
o4
v5"
trackable_list_wrapper
C
k0
l1
m2
n3
o4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec$
args?
jx
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec$
args?
jx
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

lkernel
mbias
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

nkernel
obias
$?_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

rkernel
sbias
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

tkernel
ubias
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
<
r0
s1
t2
u3"
trackable_list_wrapper
<
r0
s1
t2
u3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
*__inference_sequential_layer_call_fn_12244
*__inference_sequential_layer_call_fn_12741
*__inference_sequential_layer_call_fn_12754
*__inference_sequential_layer_call_fn_12372?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
E__inference_sequential_layer_call_and_return_conditional_losses_12820
E__inference_sequential_layer_call_and_return_conditional_losses_12900
E__inference_sequential_layer_call_and_return_conditional_losses_12389
E__inference_sequential_layer_call_and_return_conditional_losses_12406?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
 "
trackable_dict_wrapper
/
L0
?1"
trackable_list_wrapper
C
G0
H1
I2
J3
K4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
1__inference_swin_transformer_1_layer_call_fn_1176"?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
1__inference_swin_transformer_1_layer_call_fn_1623"?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_2354"?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_6078"?
???
FullArgSpec
args?
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
K
y0
z1
{2
|3
}4
?5"
trackable_list_wrapper
C
y0
z1
{2
|3
}4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec$
args?
jx
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec$
args?
jx
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

zkernel
{bias
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

|kernel
}bias
$?_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
@
?0
?1
?2
?3"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
,__inference_sequential_1_layer_call_fn_12525
,__inference_sequential_1_layer_call_fn_12913
,__inference_sequential_1_layer_call_fn_12926
,__inference_sequential_1_layer_call_fn_12653?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12992
G__inference_sequential_1_layer_call_and_return_conditional_losses_13072
G__inference_sequential_1_layer_call_and_return_conditional_losses_12670
G__inference_sequential_1_layer_call_and_return_conditional_losses_12687?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
T0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_patch_merging_layer_call_fn_2744"?
???
FullArgSpec
args?
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_patch_merging_layer_call_and_return_conditional_losses_2938"?
???
FullArgSpec
args?
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
8__inference_global_average_pooling1d_layer_call_fn_12692inputs"?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_12698inputs"?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
(__inference_dense_10_layer_call_fn_12707inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_10_layer_call_and_return_conditional_losses_12718inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
:	2StateVar
:	2StateVar
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
 "
trackable_dict_wrapper
'
v0"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_generic_user_object
 "
trackable_dict_wrapper
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_3_layer_call_fn_13081?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_3_layer_call_and_return_conditional_losses_13111?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_activation_layer_call_fn_13116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_activation_layer_call_and_return_conditional_losses_13128?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_1_layer_call_fn_13133
)__inference_dropout_1_layer_call_fn_13138?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_1_layer_call_and_return_conditional_losses_13143
D__inference_dropout_1_layer_call_and_return_conditional_losses_13155?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_4_layer_call_fn_13164?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_4_layer_call_and_return_conditional_losses_13194?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_2_layer_call_fn_13199
)__inference_dropout_2_layer_call_fn_13204?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_2_layer_call_and_return_conditional_losses_13209
D__inference_dropout_2_layer_call_and_return_conditional_losses_13221?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
H
?0
?1
?2
?3
?4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_sequential_layer_call_fn_12244dense_3_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_12741inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_12754inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_sequential_layer_call_fn_12372dense_3_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_12820inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_12900inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_12389dense_3_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_sequential_layer_call_and_return_conditional_losses_12406dense_3_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
(
?0"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_generic_user_object
 "
trackable_dict_wrapper
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_7_layer_call_fn_13230?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_7_layer_call_and_return_conditional_losses_13260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_activation_1_layer_call_fn_13265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
G__inference_activation_1_layer_call_and_return_conditional_losses_13277?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_4_layer_call_fn_13282
)__inference_dropout_4_layer_call_fn_13287?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_4_layer_call_and_return_conditional_losses_13292
D__inference_dropout_4_layer_call_and_return_conditional_losses_13304?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_8_layer_call_fn_13313?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_8_layer_call_and_return_conditional_losses_13343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_5_layer_call_fn_13348
)__inference_dropout_5_layer_call_fn_13353?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_5_layer_call_and_return_conditional_losses_13358
D__inference_dropout_5_layer_call_and_return_conditional_losses_13370?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
H
?0
?1
?2
?3
?4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_sequential_1_layer_call_fn_12525dense_7_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_1_layer_call_fn_12913inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_1_layer_call_fn_12926inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_1_layer_call_fn_12653dense_7_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12992inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_13072inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12670dense_7_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12687dense_7_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
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
?B?
'__inference_dense_3_layer_call_fn_13081inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_3_layer_call_and_return_conditional_losses_13111inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
*__inference_activation_layer_call_fn_13116inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_activation_layer_call_and_return_conditional_losses_13128inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dropout_1_layer_call_fn_13133inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
)__inference_dropout_1_layer_call_fn_13138inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_13143inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_13155inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
'__inference_dense_4_layer_call_fn_13164inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_4_layer_call_and_return_conditional_losses_13194inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dropout_2_layer_call_fn_13199inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
)__inference_dropout_2_layer_call_fn_13204inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_2_layer_call_and_return_conditional_losses_13209inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_2_layer_call_and_return_conditional_losses_13221inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
'__inference_dense_7_layer_call_fn_13230inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_7_layer_call_and_return_conditional_losses_13260inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
,__inference_activation_1_layer_call_fn_13265inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_activation_1_layer_call_and_return_conditional_losses_13277inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dropout_4_layer_call_fn_13282inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
)__inference_dropout_4_layer_call_fn_13287inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_4_layer_call_and_return_conditional_losses_13292inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_4_layer_call_and_return_conditional_losses_13304inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
'__inference_dense_8_layer_call_fn_13313inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_8_layer_call_and_return_conditional_losses_13343inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dropout_5_layer_call_fn_13348inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
)__inference_dropout_5_layer_call_fn_13353inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_5_layer_call_and_return_conditional_losses_13358inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_5_layer_call_and_return_conditional_losses_13370inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
__inference__wrapped_model_9941?)fghijlmvknopqrstuwxz{?yL|}~?????cd8?5
.?+
)?&
input_1?????????@@
? "3?0
.
dense_10"?
dense_10?????????W?
G__inference_activation_1_layer_call_and_return_conditional_losses_13277d5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
,__inference_activation_1_layer_call_fn_13265W5?2
+?(
&?#
inputs???????????
? "?????????????
E__inference_activation_layer_call_and_return_conditional_losses_13128d5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
*__inference_activation_layer_call_fn_13116W5?2
+?(
&?#
inputs???????????
? "?????????????
C__inference_dense_10_layer_call_and_return_conditional_losses_12718]cd0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????W
? |
(__inference_dense_10_layer_call_fn_12707Pcd0?-
&?#
!?
inputs??????????
? "??????????W?
B__inference_dense_3_layer_call_and_return_conditional_losses_13111grs4?1
*?'
%?"
inputs??????????@
? "+?(
!?
0???????????
? ?
'__inference_dense_3_layer_call_fn_13081Zrs4?1
*?'
%?"
inputs??????????@
? "?????????????
B__inference_dense_4_layer_call_and_return_conditional_losses_13194gtu5?2
+?(
&?#
inputs???????????
? "*?'
 ?
0??????????@
? ?
'__inference_dense_4_layer_call_fn_13164Ztu5?2
+?(
&?#
inputs???????????
? "???????????@?
B__inference_dense_7_layer_call_and_return_conditional_losses_13260i??4?1
*?'
%?"
inputs??????????@
? "+?(
!?
0???????????
? ?
'__inference_dense_7_layer_call_fn_13230\??4?1
*?'
%?"
inputs??????????@
? "?????????????
B__inference_dense_8_layer_call_and_return_conditional_losses_13343i??5?2
+?(
&?#
inputs???????????
? "*?'
 ?
0??????????@
? ?
'__inference_dense_8_layer_call_fn_13313\??5?2
+?(
&?#
inputs???????????
? "???????????@?
D__inference_dropout_1_layer_call_and_return_conditional_losses_13143h9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_13155h9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
)__inference_dropout_1_layer_call_fn_13133[9?6
/?,
&?#
inputs???????????
p 
? "?????????????
)__inference_dropout_1_layer_call_fn_13138[9?6
/?,
&?#
inputs???????????
p
? "?????????????
D__inference_dropout_2_layer_call_and_return_conditional_losses_13209f8?5
.?+
%?"
inputs??????????@
p 
? "*?'
 ?
0??????????@
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_13221f8?5
.?+
%?"
inputs??????????@
p
? "*?'
 ?
0??????????@
? ?
)__inference_dropout_2_layer_call_fn_13199Y8?5
.?+
%?"
inputs??????????@
p 
? "???????????@?
)__inference_dropout_2_layer_call_fn_13204Y8?5
.?+
%?"
inputs??????????@
p
? "???????????@?
D__inference_dropout_4_layer_call_and_return_conditional_losses_13292h9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_13304h9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
)__inference_dropout_4_layer_call_fn_13282[9?6
/?,
&?#
inputs???????????
p 
? "?????????????
)__inference_dropout_4_layer_call_fn_13287[9?6
/?,
&?#
inputs???????????
p
? "?????????????
D__inference_dropout_5_layer_call_and_return_conditional_losses_13358f8?5
.?+
%?"
inputs??????????@
p 
? "*?'
 ?
0??????????@
? ?
D__inference_dropout_5_layer_call_and_return_conditional_losses_13370f8?5
.?+
%?"
inputs??????????@
p
? "*?'
 ?
0??????????@
? ?
)__inference_dropout_5_layer_call_fn_13348Y8?5
.?+
%?"
inputs??????????@
p 
? "???????????@?
)__inference_dropout_5_layer_call_fn_13353Y8?5
.?+
%?"
inputs??????????@
p
? "???????????@?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_12698{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
8__inference_global_average_pooling1d_layer_call_fn_12692nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
@__inference_model_layer_call_and_return_conditional_losses_10967?)fghijlmvknopqrstuwxz{?yL|}~?????cd@?=
6?3
)?&
input_1?????????@@
p 

 
? "%?"
?
0?????????W
? ?
@__inference_model_layer_call_and_return_conditional_losses_11054?-??fghijlmvknopqrstuwxz{?yL|}~?????cd@?=
6?3
)?&
input_1?????????@@
p

 
? "%?"
?
0?????????W
? ?
@__inference_model_layer_call_and_return_conditional_losses_11411?)fghijlmvknopqrstuwxz{?yL|}~?????cd??<
5?2
(?%
inputs?????????@@
p 

 
? "%?"
?
0?????????W
? ?
@__inference_model_layer_call_and_return_conditional_losses_11769?-??fghijlmvknopqrstuwxz{?yL|}~?????cd??<
5?2
(?%
inputs?????????@@
p

 
? "%?"
?
0?????????W
? ?
%__inference_model_layer_call_fn_10178?)fghijlmvknopqrstuwxz{?yL|}~?????cd@?=
6?3
)?&
input_1?????????@@
p 

 
? "??????????W?
%__inference_model_layer_call_fn_10884?-??fghijlmvknopqrstuwxz{?yL|}~?????cd@?=
6?3
)?&
input_1?????????@@
p

 
? "??????????W?
%__inference_model_layer_call_fn_11206?)fghijlmvknopqrstuwxz{?yL|}~?????cd??<
5?2
(?%
inputs?????????@@
p 

 
? "??????????W?
%__inference_model_layer_call_fn_11285?-??fghijlmvknopqrstuwxz{?yL|}~?????cd??<
5?2
(?%
inputs?????????@@
p

 
? "??????????W?
I__inference_patch_embedding_layer_call_and_return_conditional_losses_2689ffgh3?0
)?&
$?!
patch??????????
? "*?'
 ?
0??????????@
? ?
.__inference_patch_embedding_layer_call_fn_1272Yfgh3?0
)?&
$?!
patch??????????
? "???????????@?
G__inference_patch_extract_layer_call_and_return_conditional_losses_1291e7?4
-?*
(?%
images?????????@@
? "*?'
 ?
0??????????
? ?
,__inference_patch_extract_layer_call_fn_3752X7?4
-?*
(?%
images?????????@@
? "????????????
G__inference_patch_merging_layer_call_and_return_conditional_losses_2938b?/?,
%?"
 ?
x??????????@
? "+?(
!?
0???????????
? ?
,__inference_patch_merging_layer_call_fn_2744U?/?,
%?"
 ?
x??????????@
? "?????????????
F__inference_random_crop_layer_call_and_return_conditional_losses_11827l;?8
1?.
(?%
inputs?????????@@
p 
? "-?*
#? 
0?????????@@
? ?
F__inference_random_crop_layer_call_and_return_conditional_losses_12000p?;?8
1?.
(?%
inputs?????????@@
p
? "-?*
#? 
0?????????@@
? ?
+__inference_random_crop_layer_call_fn_11774_;?8
1?.
(?%
inputs?????????@@
p 
? " ??????????@@?
+__inference_random_crop_layer_call_fn_11781c?;?8
1?.
(?%
inputs?????????@@
p
? " ??????????@@?
F__inference_random_flip_layer_call_and_return_conditional_losses_12016l;?8
1?.
(?%
inputs?????????@@
p 
? "-?*
#? 
0?????????@@
? ?
F__inference_random_flip_layer_call_and_return_conditional_losses_12125p?;?8
1?.
(?%
inputs?????????@@
p
? "-?*
#? 
0?????????@@
? ?
+__inference_random_flip_layer_call_fn_12005_;?8
1?.
(?%
inputs?????????@@
p 
? " ??????????@@?
+__inference_random_flip_layer_call_fn_12012c?;?8
1?.
(?%
inputs?????????@@
p
? " ??????????@@?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12670{????C?@
9?6
,?)
dense_7_input??????????@
p 

 
? "*?'
 ?
0??????????@
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12687{????C?@
9?6
,?)
dense_7_input??????????@
p

 
? "*?'
 ?
0??????????@
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_12992t????<?9
2?/
%?"
inputs??????????@
p 

 
? "*?'
 ?
0??????????@
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_13072t????<?9
2?/
%?"
inputs??????????@
p

 
? "*?'
 ?
0??????????@
? ?
,__inference_sequential_1_layer_call_fn_12525n????C?@
9?6
,?)
dense_7_input??????????@
p 

 
? "???????????@?
,__inference_sequential_1_layer_call_fn_12653n????C?@
9?6
,?)
dense_7_input??????????@
p

 
? "???????????@?
,__inference_sequential_1_layer_call_fn_12913g????<?9
2?/
%?"
inputs??????????@
p 

 
? "???????????@?
,__inference_sequential_1_layer_call_fn_12926g????<?9
2?/
%?"
inputs??????????@
p

 
? "???????????@?
E__inference_sequential_layer_call_and_return_conditional_losses_12389wrstuC?@
9?6
,?)
dense_3_input??????????@
p 

 
? "*?'
 ?
0??????????@
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_12406wrstuC?@
9?6
,?)
dense_3_input??????????@
p

 
? "*?'
 ?
0??????????@
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_12820prstu<?9
2?/
%?"
inputs??????????@
p 

 
? "*?'
 ?
0??????????@
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_12900prstu<?9
2?/
%?"
inputs??????????@
p

 
? "*?'
 ?
0??????????@
? ?
*__inference_sequential_layer_call_fn_12244jrstuC?@
9?6
,?)
dense_3_input??????????@
p 

 
? "???????????@?
*__inference_sequential_layer_call_fn_12372jrstuC?@
9?6
,?)
dense_3_input??????????@
p

 
? "???????????@?
*__inference_sequential_layer_call_fn_12741crstu<?9
2?/
%?"
inputs??????????@
p 

 
? "???????????@?
*__inference_sequential_layer_call_fn_12754crstu<?9
2?/
%?"
inputs??????????@
p

 
? "???????????@?
#__inference_signature_wrapper_11131?)fghijlmvknopqrstuwxz{?yL|}~?????cdC?@
? 
9?6
4
input_1)?&
input_1?????????@@"3?0
.
dense_10"?
dense_10?????????W?
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_2354wwxz{?yL|}~????3?0
)?&
 ?
x??????????@
p 
? "*?'
 ?
0??????????@
? ?
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_6078wwxz{?yL|}~????3?0
)?&
 ?
x??????????@
p
? "*?'
 ?
0??????????@
? ?
1__inference_swin_transformer_1_layer_call_fn_1176jwxz{?yL|}~????3?0
)?&
 ?
x??????????@
p 
? "???????????@?
1__inference_swin_transformer_1_layer_call_fn_1623jwxz{?yL|}~????3?0
)?&
 ?
x??????????@
p
? "???????????@?
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2087qijlmvknopqrstu3?0
)?&
 ?
x??????????@
p
? "*?'
 ?
0??????????@
? ?
I__inference_swin_transformer_layer_call_and_return_conditional_losses_721qijlmvknopqrstu3?0
)?&
 ?
x??????????@
p 
? "*?'
 ?
0??????????@
? ?
/__inference_swin_transformer_layer_call_fn_2650dijlmvknopqrstu3?0
)?&
 ?
x??????????@
p
? "???????????@?
/__inference_swin_transformer_layer_call_fn_6442dijlmvknopqrstu3?0
)?&
 ?
x??????????@
p 
? "???????????@