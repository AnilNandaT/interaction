і8
з%Ќ%
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
П
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
Ў
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

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
delete_old_dirsbool(
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
Г
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

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
2	
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
dtypetype
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

ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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

2	
С
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
executor_typestring Ј
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
ї
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018ѓ2
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

patch_merging/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namepatch_merging/dense_9/kernel

0patch_merging/dense_9/kernel/Read/ReadVariableOpReadVariableOppatch_merging/dense_9/kernel* 
_output_shapes
:
*
dtype0
И
.swin_transformer_1/window_attention_1/VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape
:*?
shared_name0.swin_transformer_1/window_attention_1/Variable
Б
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
shape:	@*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	@*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	@*
dtype0
В
-swin_transformer_1/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-swin_transformer_1/layer_normalization_3/beta
Ћ
Aswin_transformer_1/layer_normalization_3/beta/Read/ReadVariableOpReadVariableOp-swin_transformer_1/layer_normalization_3/beta*
_output_shapes
:@*
dtype0
Д
.swin_transformer_1/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.swin_transformer_1/layer_normalization_3/gamma
­
Bswin_transformer_1/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp.swin_transformer_1/layer_normalization_3/gamma*
_output_shapes
:@*
dtype0
М
2swin_transformer_1/window_attention_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42swin_transformer_1/window_attention_1/dense_6/bias
Е
Fswin_transformer_1/window_attention_1/dense_6/bias/Read/ReadVariableOpReadVariableOp2swin_transformer_1/window_attention_1/dense_6/bias*
_output_shapes
:@*
dtype0
Ф
4swin_transformer_1/window_attention_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*E
shared_name64swin_transformer_1/window_attention_1/dense_6/kernel
Н
Hswin_transformer_1/window_attention_1/dense_6/kernel/Read/ReadVariableOpReadVariableOp4swin_transformer_1/window_attention_1/dense_6/kernel*
_output_shapes

:@@*
dtype0
Н
2swin_transformer_1/window_attention_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*C
shared_name42swin_transformer_1/window_attention_1/dense_5/bias
Ж
Fswin_transformer_1/window_attention_1/dense_5/bias/Read/ReadVariableOpReadVariableOp2swin_transformer_1/window_attention_1/dense_5/bias*
_output_shapes	
:Р*
dtype0
Х
4swin_transformer_1/window_attention_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@Р*E
shared_name64swin_transformer_1/window_attention_1/dense_5/kernel
О
Hswin_transformer_1/window_attention_1/dense_5/kernel/Read/ReadVariableOpReadVariableOp4swin_transformer_1/window_attention_1/dense_5/kernel*
_output_shapes
:	@Р*
dtype0
Д
,swin_transformer_1/window_attention_1/weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*=
shared_name.,swin_transformer_1/window_attention_1/weight
­
@swin_transformer_1/window_attention_1/weight/Read/ReadVariableOpReadVariableOp,swin_transformer_1/window_attention_1/weight*
_output_shapes

:	*
dtype0
В
-swin_transformer_1/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-swin_transformer_1/layer_normalization_2/beta
Ћ
Aswin_transformer_1/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp-swin_transformer_1/layer_normalization_2/beta*
_output_shapes
:@*
dtype0
Д
.swin_transformer_1/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.swin_transformer_1/layer_normalization_2/gamma
­
Bswin_transformer_1/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp.swin_transformer_1/layer_normalization_2/gamma*
_output_shapes
:@*
dtype0
А
*swin_transformer/window_attention/VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape
:*;
shared_name,*swin_transformer/window_attention/Variable
Љ
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
shape:	@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	@*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	@*
dtype0
Ў
+swin_transformer/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+swin_transformer/layer_normalization_1/beta
Ї
?swin_transformer/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp+swin_transformer/layer_normalization_1/beta*
_output_shapes
:@*
dtype0
А
,swin_transformer/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,swin_transformer/layer_normalization_1/gamma
Љ
@swin_transformer/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp,swin_transformer/layer_normalization_1/gamma*
_output_shapes
:@*
dtype0
Д
.swin_transformer/window_attention/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.swin_transformer/window_attention/dense_2/bias
­
Bswin_transformer/window_attention/dense_2/bias/Read/ReadVariableOpReadVariableOp.swin_transformer/window_attention/dense_2/bias*
_output_shapes
:@*
dtype0
М
0swin_transformer/window_attention/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*A
shared_name20swin_transformer/window_attention/dense_2/kernel
Е
Dswin_transformer/window_attention/dense_2/kernel/Read/ReadVariableOpReadVariableOp0swin_transformer/window_attention/dense_2/kernel*
_output_shapes

:@@*
dtype0
Е
.swin_transformer/window_attention/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*?
shared_name0.swin_transformer/window_attention/dense_1/bias
Ў
Bswin_transformer/window_attention/dense_1/bias/Read/ReadVariableOpReadVariableOp.swin_transformer/window_attention/dense_1/bias*
_output_shapes	
:Р*
dtype0
Н
0swin_transformer/window_attention/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@Р*A
shared_name20swin_transformer/window_attention/dense_1/kernel
Ж
Dswin_transformer/window_attention/dense_1/kernel/Read/ReadVariableOpReadVariableOp0swin_transformer/window_attention/dense_1/kernel*
_output_shapes
:	@Р*
dtype0
Ќ
(swin_transformer/window_attention/weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*9
shared_name*(swin_transformer/window_attention/weight
Ѕ
<swin_transformer/window_attention/weight/Read/ReadVariableOpReadVariableOp(swin_transformer/window_attention/weight*
_output_shapes

:	*
dtype0
Њ
)swin_transformer/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)swin_transformer/layer_normalization/beta
Ѓ
=swin_transformer/layer_normalization/beta/Read/ReadVariableOpReadVariableOp)swin_transformer/layer_normalization/beta*
_output_shapes
:@*
dtype0
Ќ
*swin_transformer/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*swin_transformer/layer_normalization/gamma
Ѕ
>swin_transformer/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp*swin_transformer/layer_normalization/gamma*
_output_shapes
:@*
dtype0
Ѕ
$patch_embedding/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*5
shared_name&$patch_embedding/embedding/embeddings

8patch_embedding/embedding/embeddings/Read/ReadVariableOpReadVariableOp$patch_embedding/embedding/embeddings*
_output_shapes
:	@*
dtype0

patch_embedding/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namepatch_embedding/dense/bias

.patch_embedding/dense/bias/Read/ReadVariableOpReadVariableOppatch_embedding/dense/bias*
_output_shapes
:@*
dtype0

patch_embedding/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_namepatch_embedding/dense/kernel

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
shape:	W* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	W*
dtype0

swin_transformer_1/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameswin_transformer_1/Variable

/swin_transformer_1/Variable/Read/ReadVariableOpReadVariableOpswin_transformer_1/Variable*#
_output_shapes
:*
dtype0

NoOpNoOp
Џэ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*щь
valueоьBкь Bвь
ц
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
Ь
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
#_self_saveable_object_factories*
Ь
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_random_generator
#$_self_saveable_object_factories*
Г
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
#+_self_saveable_object_factories* 
Ю
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2proj
3	pos_embed
#4_self_saveable_object_factories*
э
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
ќ
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
Ч
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Tlinear_trans
#U_self_saveable_object_factories*
Г
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
#\_self_saveable_object_factories* 
Ы
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias
#e_self_saveable_object_factories*

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
26
27
28
29
L30
31
32
c33
d34*
џ
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
25
26
27
28
29
c30
d31*
* 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 

serving_default* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 


_generator*
* 
* 
* 
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 

Ѓtrace_0
Єtrace_1* 

Ѕtrace_0
Іtrace_1* 

Ї
_generator*
* 
* 
* 
* 

Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 

­trace_0* 

Ўtrace_0* 
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

Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

Дtrace_0* 

Еtrace_0* 
в
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses

fkernel
gbias
$М_self_saveable_object_factories*
Ь
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses
h
embeddings
$У_self_saveable_object_factories*
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

Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

Щtrace_0
Ъtrace_1* 

Ыtrace_0
Ьtrace_1* 
м
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses
	гaxis
	igamma
jbeta
$д_self_saveable_object_factories*
Њ
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses
лqkv
мdropout
	нproj

kweight
 krelative_position_bias_table
vrelative_position_index
$о_self_saveable_object_factories*
К
п	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses
$х_self_saveable_object_factories* 
м
ц	variables
чtrainable_variables
шregularization_losses
щ	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses
	ьaxis
	pgamma
qbeta
$э_self_saveable_object_factories*
И
юlayer_with_weights-0
юlayer-0
яlayer-1
№layer-2
ёlayer_with_weights-1
ёlayer-3
ђlayer-4
ѓ	variables
єtrainable_variables
ѕregularization_losses
і	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses
$љ_self_saveable_object_factories*
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
9
10
11
12
L13
14*
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
9
10
11
12*
* 

њnon_trainable_variables
ћlayers
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

џtrace_0
trace_1* 

trace_0
trace_1* 
м
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	wgamma
xbeta
$_self_saveable_object_factories*
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
qkv
dropout
	proj

yweight
 yrelative_position_bias_table
relative_position_index
$_self_saveable_object_factories*
К
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$_self_saveable_object_factories* 
м
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses
	Ђaxis
	~gamma
beta
$Ѓ_self_saveable_object_factories*
И
Єlayer_with_weights-0
Єlayer-0
Ѕlayer-1
Іlayer-2
Їlayer_with_weights-1
Їlayer-3
Јlayer-4
Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses
$Џ_self_saveable_object_factories*
nh
VARIABLE_VALUEswin_transformer_1/Variable9layer_with_weights-2/attn_mask/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*

0*
* 

Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

Еtrace_0* 

Жtrace_0* 
Щ
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses
kernel
$Н_self_saveable_object_factories*
* 
* 
* 
* 

Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

Уtrace_0* 

Фtrace_0* 
* 

c0
d1*

c0
d1*
* 

Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

Ъtrace_0* 

Ыtrace_0* 
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
2*
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

Ь0
Э1
Ю2*
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
Я
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
а
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

бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses*
* 
* 
* 

h0*

h0*
* 

жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses*
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

лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses*
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

рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses*
* 
* 
в
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses

lkernel
mbias
$ы_self_saveable_object_factories*
в
ь	variables
эtrainable_variables
юregularization_losses
я	keras_api
№__call__
+ё&call_and_return_all_conditional_losses
ђ_random_generator
$ѓ_self_saveable_object_factories* 
в
є	variables
ѕtrainable_variables
іregularization_losses
ї	keras_api
ј__call__
+љ&call_and_return_all_conditional_losses

nkernel
obias
$њ_self_saveable_object_factories*
* 
* 
* 
* 

ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
п	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses* 
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ц	variables
чtrainable_variables
шregularization_losses
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses*
* 
* 
* 
* 
в
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

rkernel
sbias
$_self_saveable_object_factories*
К
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$_self_saveable_object_factories* 
в
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator
$_self_saveable_object_factories* 
в
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses

tkernel
ubias
$Ё_self_saveable_object_factories*
в
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
Ѕ	keras_api
І__call__
+Ї&call_and_return_all_conditional_losses
Ј_random_generator
$Љ_self_saveable_object_factories* 
 
r0
s1
t2
u3*
 
r0
s1
t2
u3*
* 

Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
ѓ	variables
єtrainable_variables
ѕregularization_losses
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses*
:
Џtrace_0
Аtrace_1
Бtrace_2
Вtrace_3* 
:
Гtrace_0
Дtrace_1
Еtrace_2
Жtrace_3* 
* 

L0
1*
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

Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
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
5*
'
y0
z1
{2
|3
}4*
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
в
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses

zkernel
{bias
$Ч_self_saveable_object_factories*
в
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses
Ю_random_generator
$Я_self_saveable_object_factories* 
в
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses

|kernel
}bias
$ж_self_saveable_object_factories*
* 
* 
* 
* 

зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
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

мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses*
* 
* 
* 
* 
д
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses
kernel
	bias
$ч_self_saveable_object_factories*
К
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses
$ю_self_saveable_object_factories* 
в
я	variables
№trainable_variables
ёregularization_losses
ђ	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses
ѕ_random_generator
$і_self_saveable_object_factories* 
д
ї	variables
јtrainable_variables
љregularization_losses
њ	keras_api
ћ__call__
+ќ&call_and_return_all_conditional_losses
kernel
	bias
$§_self_saveable_object_factories*
в
ў	variables
џtrainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator
$_self_saveable_object_factories* 
$
0
1
2
3*
$
0
1
2
3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
* 

T0*
* 
* 
* 
* 
* 

0*

0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses*
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
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count
 
_fn_kwargs*
M
Ё	variables
Ђ	keras_api

Ѓtotal

Єcount
Ѕ
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

л0
м1
н2*
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

Іnon_trainable_variables
Їlayers
Јmetrics
 Љlayer_regularization_losses
Њlayer_metrics
х	variables
цtrainable_variables
чregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
ь	variables
эtrainable_variables
юregularization_losses
№__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses* 
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

Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
є	variables
ѕtrainable_variables
іregularization_losses
ј__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses*
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

Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Кtrace_0* 

Лtrace_0* 
* 
* 
* 
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Сtrace_0* 

Тtrace_0* 
* 
* 
* 
* 

Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Шtrace_0
Щtrace_1* 

Ъtrace_0
Ыtrace_1* 
* 
* 

t0
u1*

t0
u1*
* 

Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*

бtrace_0* 

вtrace_0* 
* 
* 
* 
* 

гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
І__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses* 

иtrace_0
йtrace_1* 

кtrace_0
лtrace_1* 
* 
* 
* 
,
ю0
я1
№2
ё3
ђ4*
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

0*

0
1
2*
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

мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses* 
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

цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses*
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
0
1*

0
1*
* 

ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses*

№trace_0* 

ёtrace_0* 
* 
* 
* 
* 

ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses* 

їtrace_0* 

јtrace_0* 
* 
* 
* 
* 

љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
я	variables
№trainable_variables
ёregularization_losses
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses* 

ўtrace_0
џtrace_1* 

trace_0
trace_1* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ї	variables
јtrainable_variables
љregularization_losses
ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ў	variables
џtrainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
,
Є0
Ѕ1
І2
Ї3
Ј4*
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
0
1*

	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ѓ0
Є1*

Ё	variables*
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

serving_default_input_1Placeholder*/
_output_shapes
:џџџџџџџџџ@@*
dtype0*$
shape:џџџџџџџџџ@@
ц
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1patch_embedding/dense/kernelpatch_embedding/dense/bias$patch_embedding/embedding/embeddings*swin_transformer/layer_normalization/gamma)swin_transformer/layer_normalization/beta0swin_transformer/window_attention/dense_1/kernel.swin_transformer/window_attention/dense_1/bias*swin_transformer/window_attention/Variable(swin_transformer/window_attention/weight0swin_transformer/window_attention/dense_2/kernel.swin_transformer/window_attention/dense_2/bias,swin_transformer/layer_normalization_1/gamma+swin_transformer/layer_normalization_1/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/bias.swin_transformer_1/layer_normalization_2/gamma-swin_transformer_1/layer_normalization_2/beta4swin_transformer_1/window_attention_1/dense_5/kernel2swin_transformer_1/window_attention_1/dense_5/bias.swin_transformer_1/window_attention_1/Variable,swin_transformer_1/window_attention_1/weightswin_transformer_1/Variable4swin_transformer_1/window_attention_1/dense_6/kernel2swin_transformer_1/window_attention_1/dense_6/bias.swin_transformer_1/layer_normalization_3/gamma-swin_transformer_1/layer_normalization_3/betadense_7/kerneldense_7/biasdense_8/kerneldense_8/biaspatch_merging/dense_9/kerneldense_10/kerneldense_10/bias*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџW*E
_read_only_resource_inputs'
%#	
 !"#*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_11131
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
и
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
GPU 2J 8 *'
f"R 
__inference__traced_save_13522
ї
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_13661ёѓ/
Шr
а
cond_true_10345
cond_shape_inputs;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_identityЂ'cond/crop_to_bounding_box/Assert/AssertЂ)cond/crop_to_bounding_box/Assert_1/AssertЂ)cond/crop_to_bounding_box/Assert_2/AssertЂ)cond/crop_to_bounding_box/Assert_3/AssertЂ$cond/stateful_uniform/RngReadAndSkipK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџm
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџd
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
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
ўџџџџџџџџo
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџf
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ђ
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
valueB :џџџџe
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
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
: Ъ
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
valueB:Х
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
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
valueB:Л
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :С
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
valueB:§
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
valueB:§
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
:
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.И
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.ш
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B :@
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B :@Є
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.х
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B :@
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B :@Њ
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.ч
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 Ю
,cond/crop_to_bounding_box/control_dependencyIdentitycond_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:џџџџџџџџџ@@c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ы
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:
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
valueB:е
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:н
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
value	B :@
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:ќ
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@@
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@ 
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: 2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:5 1
/
_output_shapes
:џџџџџџџџџ@@
Е
E
)__inference_dropout_4_layer_call_fn_13282

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12468f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

b
F__inference_random_flip_layer_call_and_return_conditional_losses_12016

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Н5
У
G__inference_patch_merging_layer_call_and_return_conditional_losses_2938
x=
)dense_9_tensordot_readvariableop_resource:

identityЂ dense_9/Tensordot/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   g
ReshapeReshapexReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @l
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
valueB"            і
strided_sliceStridedSliceReshape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

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
valueB"            ў
strided_slice_1StridedSliceReshape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

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
valueB"            ў
strided_slice_2StridedSliceReshape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

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
valueB"            ў
strided_slice_3StridedSliceReshape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

begin_mask	*
end_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџв
concatConcatV2strided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0strided_slice_3:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:џџџџџџџџџd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      w
	Reshape_1Reshapeconcat:output:0Reshape_1/shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource* 
_output_shapes
:
*
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
value	B : л
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
value	B : п
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
valueB: 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_9/Tensordot/transpose	TransposeReshape_1:output:0!dense_9/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџЂ
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЃ
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџi
NoOpNoOp!^dense_9/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentitydense_9/Tensordot:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:џџџџџџџџџ@: 2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
6
Н
@__inference_model_layer_call_and_return_conditional_losses_10105

inputs'
patch_embedding_10016:@#
patch_embedding_10018:@(
patch_embedding_10020:	@$
swin_transformer_10023:@$
swin_transformer_10025:@)
swin_transformer_10027:	@Р%
swin_transformer_10029:	Р(
swin_transformer_10031:	(
swin_transformer_10033:	(
swin_transformer_10035:@@$
swin_transformer_10037:@$
swin_transformer_10039:@$
swin_transformer_10041:@)
swin_transformer_10043:	@%
swin_transformer_10045:	)
swin_transformer_10047:	@$
swin_transformer_10049:@&
swin_transformer_1_10052:@&
swin_transformer_1_10054:@+
swin_transformer_1_10056:	@Р'
swin_transformer_1_10058:	Р*
swin_transformer_1_10060:	*
swin_transformer_1_10062:	/
swin_transformer_1_10064:*
swin_transformer_1_10066:@@&
swin_transformer_1_10068:@&
swin_transformer_1_10070:@&
swin_transformer_1_10072:@+
swin_transformer_1_10074:	@'
swin_transformer_1_10076:	+
swin_transformer_1_10078:	@&
swin_transformer_1_10080:@'
patch_merging_10083:
!
dense_10_10099:	W
dense_10_10101:W
identityЂ dense_10/StatefulPartitionedCallЂ'patch_embedding/StatefulPartitionedCallЂ%patch_merging/StatefulPartitionedCallЂ(swin_transformer/StatefulPartitionedCallЂ*swin_transformer_1/StatefulPartitionedCallХ
random_crop/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10007у
random_flip/PartitionedCallPartitionedCall$random_crop/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10013У
patch_extract/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9773Є
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_10016patch_embedding_10018patch_embedding_10020*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9785а
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_10023swin_transformer_10025swin_transformer_10027swin_transformer_10029swin_transformer_10031swin_transformer_10033swin_transformer_10035swin_transformer_10037swin_transformer_10039swin_transformer_10041swin_transformer_10043swin_transformer_10045swin_transformer_10047swin_transformer_10049*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9825
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_10052swin_transformer_1_10054swin_transformer_1_10056swin_transformer_1_10058swin_transformer_1_10060swin_transformer_1_10062swin_transformer_1_10064swin_transformer_1_10066swin_transformer_1_10068swin_transformer_1_10070swin_transformer_1_10072swin_transformer_1_10074swin_transformer_1_10076swin_transformer_1_10078swin_transformer_1_10080*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9889ќ
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_10083*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9927џ
(global_average_pooling1d/PartitionedCallPartitionedCall.patch_merging/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951
 dense_10/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_10_10099dense_10_10101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџW*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_10098x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџW
NoOpNoOp!^dense_10/StatefulPartitionedCall(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
э9
Ь
@__inference_model_layer_call_and_return_conditional_losses_11054
input_1
random_crop_10970:	
random_flip_10973:	'
patch_embedding_10977:@#
patch_embedding_10979:@(
patch_embedding_10981:	@$
swin_transformer_10984:@$
swin_transformer_10986:@)
swin_transformer_10988:	@Р%
swin_transformer_10990:	Р(
swin_transformer_10992:	(
swin_transformer_10994:	(
swin_transformer_10996:@@$
swin_transformer_10998:@$
swin_transformer_11000:@$
swin_transformer_11002:@)
swin_transformer_11004:	@%
swin_transformer_11006:	)
swin_transformer_11008:	@$
swin_transformer_11010:@&
swin_transformer_1_11013:@&
swin_transformer_1_11015:@+
swin_transformer_1_11017:	@Р'
swin_transformer_1_11019:	Р*
swin_transformer_1_11021:	*
swin_transformer_1_11023:	/
swin_transformer_1_11025:*
swin_transformer_1_11027:@@&
swin_transformer_1_11029:@&
swin_transformer_1_11031:@&
swin_transformer_1_11033:@+
swin_transformer_1_11035:	@'
swin_transformer_1_11037:	+
swin_transformer_1_11039:	@&
swin_transformer_1_11041:@'
patch_merging_11044:
!
dense_10_11048:	W
dense_10_11050:W
identityЂ dense_10/StatefulPartitionedCallЂ'patch_embedding/StatefulPartitionedCallЂ%patch_merging/StatefulPartitionedCallЂ#random_crop/StatefulPartitionedCallЂ#random_flip/StatefulPartitionedCallЂ(swin_transformer/StatefulPartitionedCallЂ*swin_transformer_1/StatefulPartitionedCallъ
#random_crop/StatefulPartitionedCallStatefulPartitionedCallinput_1random_crop_10970*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10491
#random_flip/StatefulPartitionedCallStatefulPartitionedCall,random_crop/StatefulPartitionedCall:output:0random_flip_10973*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10305Ы
patch_extract/PartitionedCallPartitionedCall,random_flip/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9773Є
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_10977patch_embedding_10979patch_embedding_10981*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9785б
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_10984swin_transformer_10986swin_transformer_10988swin_transformer_10990swin_transformer_10992swin_transformer_10994swin_transformer_10996swin_transformer_10998swin_transformer_11000swin_transformer_11002swin_transformer_11004swin_transformer_11006swin_transformer_11008swin_transformer_11010*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_restored_function_body_10622
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_11013swin_transformer_1_11015swin_transformer_1_11017swin_transformer_1_11019swin_transformer_1_11021swin_transformer_1_11023swin_transformer_1_11025swin_transformer_1_11027swin_transformer_1_11029swin_transformer_1_11031swin_transformer_1_11033swin_transformer_1_11035swin_transformer_1_11037swin_transformer_1_11039swin_transformer_1_11041*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_restored_function_body_10686ќ
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_11044*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9927џ
(global_average_pooling1d/PartitionedCallPartitionedCall.patch_merging/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951
 dense_10/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_10_11048dense_10_11050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџW*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_10098x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџWп
NoOpNoOp!^dense_10/StatefulPartitionedCall(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall$^random_crop/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2J
#random_crop/StatefulPartitionedCall#random_crop/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@@
!
_user_specified_name	input_1
Б
E
)__inference_dropout_2_layer_call_fn_13199

inputs
identityД
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12230e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

C
'__inference_restored_function_body_9773

images
identity
PartitionedCallPartitionedCallimages*
Tin
2*
Tout
2*,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_patch_extract_layer_call_and_return_conditional_losses_1291e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameimages
ы
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_12230

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџ@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Й
ш
6map_while_stateless_random_flip_left_right_false_12092u
qmap_while_stateless_random_flip_left_right_identity_map_while_stateless_random_flip_left_right_control_dependency7
3map_while_stateless_random_flip_left_right_identityп
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
Ђ
ѕ
 random_flip_map_while_cond_11595<
8random_flip_map_while_random_flip_map_while_loop_counter7
3random_flip_map_while_random_flip_map_strided_slice%
!random_flip_map_while_placeholder'
#random_flip_map_while_placeholder_1<
8random_flip_map_while_less_random_flip_map_strided_sliceS
Orandom_flip_map_while_random_flip_map_while_cond_11595___redundant_placeholder0S
Orandom_flip_map_while_random_flip_map_while_cond_11595___redundant_placeholder1"
random_flip_map_while_identity
 
random_flip/map/while/LessLess!random_flip_map_while_placeholder8random_flip_map_while_less_random_flip_map_strided_slice*
T0*
_output_shapes
: Д
random_flip/map/while/Less_1Less8random_flip_map_while_random_flip_map_while_loop_counter3random_flip_map_while_random_flip_map_strided_slice*
T0*
_output_shapes
: 
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
	
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
:џџџџџџџџџP
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?m
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџY
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџO

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџe

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџ\
IdentityIdentityGelu/mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
и
ћ
B__inference_dense_7_layer_call_and_return_conditional_losses_13260

inputs4
!tensordot_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџe
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѕЋ

!__inference__traced_restore_13661
file_prefixC
,assignvariableop_swin_transformer_1_variable:5
"assignvariableop_1_dense_10_kernel:	W.
 assignvariableop_2_dense_10_bias:WA
/assignvariableop_3_patch_embedding_dense_kernel:@;
-assignvariableop_4_patch_embedding_dense_bias:@J
7assignvariableop_5_patch_embedding_embedding_embeddings:	@K
=assignvariableop_6_swin_transformer_layer_normalization_gamma:@J
<assignvariableop_7_swin_transformer_layer_normalization_beta:@M
;assignvariableop_8_swin_transformer_window_attention_weight:	V
Cassignvariableop_9_swin_transformer_window_attention_dense_1_kernel:	@РQ
Bassignvariableop_10_swin_transformer_window_attention_dense_1_bias:	РV
Dassignvariableop_11_swin_transformer_window_attention_dense_2_kernel:@@P
Bassignvariableop_12_swin_transformer_window_attention_dense_2_bias:@N
@assignvariableop_13_swin_transformer_layer_normalization_1_gamma:@M
?assignvariableop_14_swin_transformer_layer_normalization_1_beta:@5
"assignvariableop_15_dense_3_kernel:	@/
 assignvariableop_16_dense_3_bias:	5
"assignvariableop_17_dense_4_kernel:	@.
 assignvariableop_18_dense_4_bias:@P
>assignvariableop_19_swin_transformer_window_attention_variable:	P
Bassignvariableop_20_swin_transformer_1_layer_normalization_2_gamma:@O
Aassignvariableop_21_swin_transformer_1_layer_normalization_2_beta:@R
@assignvariableop_22_swin_transformer_1_window_attention_1_weight:	[
Hassignvariableop_23_swin_transformer_1_window_attention_1_dense_5_kernel:	@РU
Fassignvariableop_24_swin_transformer_1_window_attention_1_dense_5_bias:	РZ
Hassignvariableop_25_swin_transformer_1_window_attention_1_dense_6_kernel:@@T
Fassignvariableop_26_swin_transformer_1_window_attention_1_dense_6_bias:@P
Bassignvariableop_27_swin_transformer_1_layer_normalization_3_gamma:@O
Aassignvariableop_28_swin_transformer_1_layer_normalization_3_beta:@5
"assignvariableop_29_dense_7_kernel:	@/
 assignvariableop_30_dense_7_bias:	5
"assignvariableop_31_dense_8_kernel:	@.
 assignvariableop_32_dense_8_bias:@T
Bassignvariableop_33_swin_transformer_1_window_attention_1_variable:	D
0assignvariableop_34_patch_merging_dense_9_kernel:
,
assignvariableop_35_statevar_1:	*
assignvariableop_36_statevar:	%
assignvariableop_37_total_2: %
assignvariableop_38_count_2: %
assignvariableop_39_total_1: %
assignvariableop_40_count_1: #
assignvariableop_41_total: #
assignvariableop_42_count: 
identity_44ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ј
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*Ю
valueФBС,B9layer_with_weights-2/attn_mask/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEBJlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHШ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B §
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp,assignvariableop_swin_transformer_1_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_10_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_10_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp/assignvariableop_3_patch_embedding_dense_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp-assignvariableop_4_patch_embedding_dense_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_5AssignVariableOp7assignvariableop_5_patch_embedding_embedding_embeddingsIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_6AssignVariableOp=assignvariableop_6_swin_transformer_layer_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_7AssignVariableOp<assignvariableop_7_swin_transformer_layer_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_8AssignVariableOp;assignvariableop_8_swin_transformer_window_attention_weightIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_9AssignVariableOpCassignvariableop_9_swin_transformer_window_attention_dense_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_10AssignVariableOpBassignvariableop_10_swin_transformer_window_attention_dense_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_11AssignVariableOpDassignvariableop_11_swin_transformer_window_attention_dense_2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_12AssignVariableOpBassignvariableop_12_swin_transformer_window_attention_dense_2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_13AssignVariableOp@assignvariableop_13_swin_transformer_layer_normalization_1_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_14AssignVariableOp?assignvariableop_14_swin_transformer_layer_normalization_1_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp assignvariableop_16_dense_3_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_4_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_4_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0	*
_output_shapes
:Џ
AssignVariableOp_19AssignVariableOp>assignvariableop_19_swin_transformer_window_attention_variableIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_20AssignVariableOpBassignvariableop_20_swin_transformer_1_layer_normalization_2_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_21AssignVariableOpAassignvariableop_21_swin_transformer_1_layer_normalization_2_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_22AssignVariableOp@assignvariableop_22_swin_transformer_1_window_attention_1_weightIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_23AssignVariableOpHassignvariableop_23_swin_transformer_1_window_attention_1_dense_5_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_24AssignVariableOpFassignvariableop_24_swin_transformer_1_window_attention_1_dense_5_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_25AssignVariableOpHassignvariableop_25_swin_transformer_1_window_attention_1_dense_6_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_26AssignVariableOpFassignvariableop_26_swin_transformer_1_window_attention_1_dense_6_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_27AssignVariableOpBassignvariableop_27_swin_transformer_1_layer_normalization_3_gammaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_28AssignVariableOpAassignvariableop_28_swin_transformer_1_layer_normalization_3_betaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp"assignvariableop_29_dense_7_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp assignvariableop_30_dense_7_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp"assignvariableop_31_dense_8_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp assignvariableop_32_dense_8_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_33AssignVariableOpBassignvariableop_33_swin_transformer_1_window_attention_1_variableIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_34AssignVariableOp0assignvariableop_34_patch_merging_dense_9_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_35AssignVariableOpassignvariableop_35_statevar_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_36AssignVariableOpassignvariableop_36_statevarIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_2Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: ю
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
е

'__inference_dense_8_layer_call_fn_13313

inputs
unknown:	@
	unknown_0:@
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_12500t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н5
У
G__inference_patch_merging_layer_call_and_return_conditional_losses_2738
x=
)dense_9_tensordot_readvariableop_resource:

identityЂ dense_9/Tensordot/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   g
ReshapeReshapexReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @l
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
valueB"            і
strided_sliceStridedSliceReshape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

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
valueB"            ў
strided_slice_1StridedSliceReshape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

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
valueB"            ў
strided_slice_2StridedSliceReshape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

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
valueB"            ў
strided_slice_3StridedSliceReshape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@*

begin_mask	*
end_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџв
concatConcatV2strided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0strided_slice_3:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:џџџџџџџџџd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      w
	Reshape_1Reshapeconcat:output:0Reshape_1/shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource* 
_output_shapes
:
*
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
value	B : л
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
value	B : п
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
valueB: 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_9/Tensordot/transpose	TransposeReshape_1:output:0!dense_9/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџЂ
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЃ
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџi
NoOpNoOp!^dense_9/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentitydense_9/Tensordot:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:џџџџџџџџџ@: 2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
з
п
/__inference_swin_transformer_layer_call_fn_2650
x
unknown:@
	unknown_0:@
	unknown_1:	@Р
	unknown_2:	Р
	unknown_3:	
	unknown_4:	
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:	@

unknown_10:	

unknown_11:	@

unknown_12:@
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2631`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
Шr
а
cond_true_11854
cond_shape_inputs;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_identityЂ'cond/crop_to_bounding_box/Assert/AssertЂ)cond/crop_to_bounding_box/Assert_1/AssertЂ)cond/crop_to_bounding_box/Assert_2/AssertЂ)cond/crop_to_bounding_box/Assert_3/AssertЂ$cond/stateful_uniform/RngReadAndSkipK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџm
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџd
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
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
ўџџџџџџџџo
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџf
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ђ
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
valueB :џџџџe
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
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
: Ъ
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
valueB:Х
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
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
valueB:Л
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :С
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
valueB:§
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
valueB:§
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
:
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.И
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.ш
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 a
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B :@
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: g
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B :@Є
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.х
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 c
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B :@
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: i
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B :@Њ
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: 
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.ч
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 Ю
,cond/crop_to_bounding_box/control_dependencyIdentitycond_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:џџџџџџџџџ@@c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Ы
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:
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
valueB:е
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
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
valueB:н
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
value	B :@
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:ќ
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@@
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@ 
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: 2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:5 1
/
_output_shapes
:џџџџџџџџџ@@
Ф
з
*__inference_sequential_layer_call_fn_12372
dense_3_input
unknown:	@
	unknown_0:	
	unknown_1:	@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_3_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12348t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ@
'
_user_specified_namedense_3_input
я
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_12468

inputs

identity_1T
IdentityIdentityinputs*
T0*-
_output_shapes
:џџџџџџџџџa

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь[

@__inference_model_layer_call_and_return_conditional_losses_11411

inputs'
patch_embedding_11331:@#
patch_embedding_11333:@(
patch_embedding_11335:	@$
swin_transformer_11338:@$
swin_transformer_11340:@)
swin_transformer_11342:	@Р%
swin_transformer_11344:	Р(
swin_transformer_11346:	(
swin_transformer_11348:	(
swin_transformer_11350:@@$
swin_transformer_11352:@$
swin_transformer_11354:@$
swin_transformer_11356:@)
swin_transformer_11358:	@%
swin_transformer_11360:	)
swin_transformer_11362:	@$
swin_transformer_11364:@&
swin_transformer_1_11367:@&
swin_transformer_1_11369:@+
swin_transformer_1_11371:	@Р'
swin_transformer_1_11373:	Р*
swin_transformer_1_11375:	*
swin_transformer_1_11377:	/
swin_transformer_1_11379:*
swin_transformer_1_11381:@@&
swin_transformer_1_11383:@&
swin_transformer_1_11385:@&
swin_transformer_1_11387:@+
swin_transformer_1_11389:	@'
swin_transformer_1_11391:	+
swin_transformer_1_11393:	@&
swin_transformer_1_11395:@'
patch_merging_11398:
:
'dense_10_matmul_readvariableop_resource:	W6
(dense_10_biasadd_readvariableop_resource:W
identityЂdense_10/BiasAdd/ReadVariableOpЂdense_10/MatMul/ReadVariableOpЂ'patch_embedding/StatefulPartitionedCallЂ%patch_merging/StatefulPartitionedCallЂ(swin_transformer/StatefulPartitionedCallЂ*swin_transformer_1/StatefulPartitionedCallG
random_crop/ShapeShapeinputs*
T0*
_output_shapes
:r
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџt
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџk
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
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
ўџџџџџџџџv
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџm
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
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
 *  Bu
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
 *  B{
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
value	B : Г
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
џџџџџџџџџ`
random_crop/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџН
random_crop/stack_1Packrandom_crop/stack_1/0:output:0random_crop/Minimum:z:0random_crop/Minimum_1:z:0random_crop/stack_1/3:output:0*
N*
T0*
_output_shapes
:Ќ
random_crop/SliceSliceinputsrandom_crop/stack:output:0random_crop/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"џџџџџџџџџ@@џџџџџџџџџh
random_crop/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   Х
!random_crop/resize/ResizeBilinearResizeBilinearrandom_crop/Slice:output:0 random_crop/resize/size:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
half_pixel_centers(б
patch_extract/PartitionedCallPartitionedCall2random_crop/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9773Є
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_11331patch_embedding_11333patch_embedding_11335*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9785а
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_11338swin_transformer_11340swin_transformer_11342swin_transformer_11344swin_transformer_11346swin_transformer_11348swin_transformer_11350swin_transformer_11352swin_transformer_11354swin_transformer_11356swin_transformer_11358swin_transformer_11360swin_transformer_11362swin_transformer_11364*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9825
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_11367swin_transformer_1_11369swin_transformer_1_11371swin_transformer_1_11373swin_transformer_1_11375swin_transformer_1_11377swin_transformer_1_11379swin_transformer_1_11381swin_transformer_1_11383swin_transformer_1_11385swin_transformer_1_11387swin_transformer_1_11389swin_transformer_1_11391swin_transformer_1_11393swin_transformer_1_11395*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9889ќ
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_11398*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9927q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Т
global_average_pooling1d/MeanMean.patch_merging/StatefulPartitionedCall:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	W*
dtype0
dense_10/MatMulMatMul&global_average_pooling1d/Mean:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџW
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:W*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџWh
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџWi
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџWГ
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
єж

L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_2354
xI
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@O
<window_attention_1_dense_5_tensordot_readvariableop_resource:	@РI
:window_attention_1_dense_5_biasadd_readvariableop_resource:	РF
4window_attention_1_reshape_1_readvariableop_resource:	4
"window_attention_1_gather_resource:	N
7window_attention_1_expanddims_1_readvariableop_resource:N
<window_attention_1_dense_6_tensordot_readvariableop_resource:@@H
:window_attention_1_dense_6_biasadd_readvariableop_resource:@I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_3_batchnorm_readvariableop_resource:@I
6sequential_1_dense_7_tensordot_readvariableop_resource:	@C
4sequential_1_dense_7_biasadd_readvariableop_resource:	I
6sequential_1_dense_8_tensordot_readvariableop_resource:	@B
4sequential_1_dense_8_biasadd_readvariableop_resource:@
identityЂ.layer_normalization_2/batchnorm/ReadVariableOpЂ2layer_normalization_2/batchnorm/mul/ReadVariableOpЂ.layer_normalization_3/batchnorm/ReadVariableOpЂ2layer_normalization_3/batchnorm/mul/ReadVariableOpЂ+sequential_1/dense_7/BiasAdd/ReadVariableOpЂ-sequential_1/dense_7/Tensordot/ReadVariableOpЂ+sequential_1/dense_8/BiasAdd/ReadVariableOpЂ-sequential_1/dense_8/Tensordot/ReadVariableOpЂ.window_attention_1/ExpandDims_1/ReadVariableOpЂwindow_attention_1/GatherЂ+window_attention_1/Reshape_1/ReadVariableOpЂ1window_attention_1/dense_5/BiasAdd/ReadVariableOpЂ3window_attention_1/dense_5/Tensordot/ReadVariableOpЂ1window_attention_1/dense_6/BiasAdd/ReadVariableOpЂ3window_attention_1/dense_6/Tensordot/ReadVariableOp~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Д
"layer_normalization_2/moments/meanMeanx=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencex3layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_2/batchnorm/mul_1Mulx'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   
ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @[

Roll/shiftConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџZ
	Roll/axisConst*
_output_shapes
:*
dtype0*
valueB"      
RollRollReshape:output:0Roll/shift:output:0Roll/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:џџџџџџџџџ  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_1ReshapeRoll:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Б
3window_attention_1/dense_5/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@Р*
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
value	B : Ї
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
value	B : Ћ
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
valueB: П
)window_attention_1/dense_5/Tensordot/ProdProd6window_attention_1/dense_5/Tensordot/GatherV2:output:03window_attention_1/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Х
+window_attention_1/dense_5/Tensordot/Prod_1Prod8window_attention_1/dense_5/Tensordot/GatherV2_1:output:05window_attention_1/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention_1/dense_5/Tensordot/concatConcatV22window_attention_1/dense_5/Tensordot/free:output:02window_attention_1/dense_5/Tensordot/axes:output:09window_attention_1/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ъ
*window_attention_1/dense_5/Tensordot/stackPack2window_attention_1/dense_5/Tensordot/Prod:output:04window_attention_1/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Л
.window_attention_1/dense_5/Tensordot/transpose	TransposeReshape_3:output:04window_attention_1/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@л
,window_attention_1/dense_5/Tensordot/ReshapeReshape2window_attention_1/dense_5/Tensordot/transpose:y:03window_attention_1/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџм
+window_attention_1/dense_5/Tensordot/MatMulMatMul5window_attention_1/dense_5/Tensordot/Reshape:output:0;window_attention_1/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџРw
,window_attention_1/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Рt
2window_attention_1/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-window_attention_1/dense_5/Tensordot/concat_1ConcatV26window_attention_1/dense_5/Tensordot/GatherV2:output:05window_attention_1/dense_5/Tensordot/Const_2:output:0;window_attention_1/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:е
$window_attention_1/dense_5/TensordotReshape5window_attention_1/dense_5/Tensordot/MatMul:product:06window_attention_1/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџРЉ
1window_attention_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype0Ю
"window_attention_1/dense_5/BiasAddBiasAdd-window_attention_1/dense_5/Tensordot:output:09window_attention_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџР}
 window_attention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Л
window_attention_1/ReshapeReshape+window_attention_1/dense_5/BiasAdd:output:0)window_attention_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ~
!window_attention_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                И
window_attention_1/transpose	Transpose#window_attention_1/Reshape:output:0*window_attention_1/transpose/perm:output:0*
T0*3
_output_shapes!
:џџџџџџџџџp
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
valueB:Ш
 window_attention_1/strided_sliceStridedSlice window_attention_1/transpose:y:0/window_attention_1/strided_slice/stack:output:01window_attention_1/strided_slice/stack_1:output:01window_attention_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:а
"window_attention_1/strided_slice_1StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_1/stack:output:03window_attention_1/strided_slice_1/stack_1:output:03window_attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:а
"window_attention_1/strided_slice_2StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_2/stack:output:03window_attention_1/strided_slice_2/stack_1:output:03window_attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask]
window_attention_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ>Ѕ
window_attention_1/mulMul)window_attention_1/strided_slice:output:0!window_attention_1/mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|
#window_attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Р
window_attention_1/transpose_1	Transpose+window_attention_1/strided_slice_1:output:0,window_attention_1/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџЄ
window_attention_1/matmulBatchMatMulV2window_attention_1/mul:z:0"window_attention_1/transpose_1:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 
+window_attention_1/Reshape_1/ReadVariableOpReadVariableOp4window_attention_1_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	u
"window_attention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЎ
window_attention_1/Reshape_1Reshape3window_attention_1/Reshape_1/ReadVariableOp:value:0+window_attention_1/Reshape_1/shape:output:0*
T0	*
_output_shapes
:Г
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
valueB"      џџџџЇ
window_attention_1/Reshape_2Reshape$window_attention_1/Identity:output:0+window_attention_1/Reshape_2/shape:output:0*
T0*"
_output_shapes
:x
#window_attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
window_attention_1/transpose_2	Transpose%window_attention_1/Reshape_2:output:0,window_attention_1/transpose_2/perm:output:0*
T0*"
_output_shapes
:c
!window_attention_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ќ
window_attention_1/ExpandDims
ExpandDims"window_attention_1/transpose_2:y:0*window_attention_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:Ѕ
window_attention_1/addAddV2"window_attention_1/matmul:output:0&window_attention_1/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџЋ
.window_attention_1/ExpandDims_1/ReadVariableOpReadVariableOp7window_attention_1_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype0e
#window_attention_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :Х
window_attention_1/ExpandDims_1
ExpandDims6window_attention_1/ExpandDims_1/ReadVariableOp:value:0,window_attention_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:e
#window_attention_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : Л
window_attention_1/ExpandDims_2
ExpandDims(window_attention_1/ExpandDims_1:output:0,window_attention_1/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:
window_attention_1/CastCast(window_attention_1/ExpandDims_2:output:0*

DstT0*

SrcT0*+
_output_shapes
:
"window_attention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Џ
window_attention_1/Reshape_3Reshapewindow_attention_1/add:z:0+window_attention_1/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџЄ
window_attention_1/add_1AddV2%window_attention_1/Reshape_3:output:0window_attention_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџ{
"window_attention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         Ќ
window_attention_1/Reshape_4Reshapewindow_attention_1/add_1:z:0+window_attention_1/Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
window_attention_1/SoftmaxSoftmax%window_attention_1/Reshape_4:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
%window_attention_1/dropout_3/IdentityIdentity$window_attention_1/Softmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџУ
window_attention_1/matmul_1BatchMatMulV2.window_attention_1/dropout_3/Identity:output:0+window_attention_1/strided_slice_2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|
#window_attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Й
window_attention_1/transpose_3	Transpose$window_attention_1/matmul_1:output:0,window_attention_1/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
"window_attention_1/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   Ў
window_attention_1/Reshape_5Reshape"window_attention_1/transpose_3:y:0+window_attention_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@А
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
value	B : Ї
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
value	B : Ћ
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
valueB: П
)window_attention_1/dense_6/Tensordot/ProdProd6window_attention_1/dense_6/Tensordot/GatherV2:output:03window_attention_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Х
+window_attention_1/dense_6/Tensordot/Prod_1Prod8window_attention_1/dense_6/Tensordot/GatherV2_1:output:05window_attention_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention_1/dense_6/Tensordot/concatConcatV22window_attention_1/dense_6/Tensordot/free:output:02window_attention_1/dense_6/Tensordot/axes:output:09window_attention_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ъ
*window_attention_1/dense_6/Tensordot/stackPack2window_attention_1/dense_6/Tensordot/Prod:output:04window_attention_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ю
.window_attention_1/dense_6/Tensordot/transpose	Transpose%window_attention_1/Reshape_5:output:04window_attention_1/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@л
,window_attention_1/dense_6/Tensordot/ReshapeReshape2window_attention_1/dense_6/Tensordot/transpose:y:03window_attention_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџл
+window_attention_1/dense_6/Tensordot/MatMulMatMul5window_attention_1/dense_6/Tensordot/Reshape:output:0;window_attention_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
,window_attention_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2window_attention_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-window_attention_1/dense_6/Tensordot/concat_1ConcatV26window_attention_1/dense_6/Tensordot/GatherV2:output:05window_attention_1/dense_6/Tensordot/Const_2:output:0;window_attention_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:д
$window_attention_1/dense_6/TensordotReshape5window_attention_1/dense_6/Tensordot/MatMul:product:06window_attention_1/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ј
1window_attention_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Э
"window_attention_1/dense_6/BiasAddBiasAdd-window_attention_1/dense_6/Tensordot:output:09window_attention_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@
'window_attention_1/dropout_3/Identity_1Identity+window_attention_1/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   
	Reshape_4Reshape0window_attention_1/dropout_3/Identity_1:output:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @]
Roll_1/shiftConst*
_output_shapes
:*
dtype0*
valueB"      \
Roll_1/axisConst*
_output_shapes
:*
dtype0*
valueB"      Є
Roll_1RollReshape_6:output:0Roll_1/shift:output:0Roll_1/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:џџџџџџџџџ  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   v
	Reshape_7ReshapeRoll_1:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@S
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
valueB:
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
value	B :д
 drop_path_1/random_uniform/shapePack"drop_path_1/strided_slice:output:0+drop_path_1/random_uniform/shape/1:output:0+drop_path_1/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:Ї
(drop_path_1/random_uniform/RandomUniformRandomUniform)drop_path_1/random_uniform/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0V
drop_path_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path_1/addAddV2drop_path_1/add/x:output:01drop_path_1/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџe
drop_path_1/FloorFloordrop_path_1/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
drop_path_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path_1/truedivRealDivReshape_7:output:0drop_path_1/truediv/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@}
drop_path_1/mulMuldrop_path_1/truediv:z:0drop_path_1/Floor:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@[
addAddV2xdrop_path_1/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:К
"layer_normalization_3/moments/meanMeanadd:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЙ
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_3/batchnorm/mul_1Muladd:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ѕ
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: ­
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Г
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : №
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:И
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ч
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Щ
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЪ
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџq
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ћ
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ф
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Н
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџi
$sequential_1/activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?З
"sequential_1/activation_1/Gelu/mulMul-sequential_1/activation_1/Gelu/mul/x:output:0%sequential_1/dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџj
%sequential_1/activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?Р
&sequential_1/activation_1/Gelu/truedivRealDiv%sequential_1/dense_7/BiasAdd:output:0.sequential_1/activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
"sequential_1/activation_1/Gelu/ErfErf*sequential_1/activation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџi
$sequential_1/activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?К
"sequential_1/activation_1/Gelu/addAddV2-sequential_1/activation_1/Gelu/add/x:output:0&sequential_1/activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџГ
$sequential_1/activation_1/Gelu/mul_1Mul&sequential_1/activation_1/Gelu/mul:z:0&sequential_1/activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџ
sequential_1/dropout_4/IdentityIdentity(sequential_1/activation_1/Gelu/mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџЅ
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: ­
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Г
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : №
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:И
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ч
(sequential_1/dense_8/Tensordot/transpose	Transpose(sequential_1/dropout_4/Identity:output:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџЩ
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЩ
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ћ
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:У
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0М
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
sequential_1/dropout_5/IdentityIdentity%sequential_1/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@k
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
valueB:
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
value	B :м
"drop_path_1/random_uniform_1/shapePack$drop_path_1/strided_slice_1:output:0-drop_path_1/random_uniform_1/shape/1:output:0-drop_path_1/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ћ
*drop_path_1/random_uniform_1/RandomUniformRandomUniform+drop_path_1/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0X
drop_path_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?Ѓ
drop_path_1/add_1AddV2drop_path_1/add_1/x:output:03drop_path_1/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџi
drop_path_1/Floor_1Floordrop_path_1/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ\
drop_path_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?Ѓ
drop_path_1/truediv_1RealDiv(sequential_1/dropout_5/Identity:output:0 drop_path_1/truediv_1/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
drop_path_1/mul_1Muldrop_path_1/truediv_1:z:0drop_path_1/Floor_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@e
add_1AddV2add:z:0drop_path_1/mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@
NoOpNoOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp/^window_attention_1/ExpandDims_1/ReadVariableOp^window_attention_1/Gather,^window_attention_1/Reshape_1/ReadVariableOp2^window_attention_1/dense_5/BiasAdd/ReadVariableOp4^window_attention_1/dense_5/Tensordot/ReadVariableOp2^window_attention_1/dense_6/BiasAdd/ReadVariableOp4^window_attention_1/dense_6/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ@: : : : : : : : : : : : : : : 2`
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
:џџџџџџџџџ@

_user_specified_namex
Й
ш
6map_while_stateless_random_flip_left_right_false_10272u
qmap_while_stateless_random_flip_left_right_identity_map_while_stateless_random_flip_left_right_control_dependency7
3map_while_stateless_random_flip_left_right_identityп
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
лV
з
E__inference_sequential_layer_call_and_return_conditional_losses_12900

inputs<
)dense_3_tensordot_readvariableop_resource:	@6
'dense_3_biasadd_readvariableop_resource:	<
)dense_4_tensordot_readvariableop_resource:	@5
'dense_4_biasadd_readvariableop_resource:@
identityЂdense_3/BiasAdd/ReadVariableOpЂ dense_3/Tensordot/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂ dense_4/Tensordot/ReadVariableOp
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : л
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
value	B : п
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
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/transpose	Transposeinputs!dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЃ
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџZ
activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
activation/Gelu/mulMulactivation/Gelu/mul/x:output:0dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџ[
activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?
activation/Gelu/truedivRealDivdense_3/BiasAdd:output:0activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџo
activation/Gelu/ErfErfactivation/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџZ
activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
activation/Gelu/addAddV2activation/Gelu/add/x:output:0activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџ
activation/Gelu/mul_1Mulactivation/Gelu/mul:z:0activation/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџ\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?
dropout_1/dropout/MulMulactivation/Gelu/mul_1:z:0 dropout_1/dropout/Const:output:0*
T0*-
_output_shapes
:џџџџџџџџџ`
dropout_1/dropout/ShapeShapeactivation/Gelu/mul_1:z:0*
T0*
_output_shapes
:І
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Ъ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:џџџџџџџџџ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*-
_output_shapes
:џџџџџџџџџ
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : л
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
value	B : п
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
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
dense_4/Tensordot/transpose	Transposedropout_1/dropout/Mul_1:z:0!dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџЂ
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЂ
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?
dropout_2/dropout/MulMuldense_4/BiasAdd:output:0 dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@_
dropout_2/dropout/ShapeShapedense_4/BiasAdd:output:0*
T0*
_output_shapes
:Ѕ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Щ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ@
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@o
IdentityIdentitydropout_2/dropout/Mul_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@Ю
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
У
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
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
ExtractImagePatchesExtractImagePatchesimages*
T0*/
_output_shapes
:џџџџџџџџџ  *
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
B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameimages


c
D__inference_dropout_2_layer_call_and_return_conditional_losses_13221

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Ћ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
 e
л
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
Grandom_flip_map_while_stateful_uniform_full_int_rngreadandskip_resource:	Ђ>random_flip/map/while/stateful_uniform_full_int/RngReadAndSkip
Grandom_flip/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      ё
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
valueB: н
4random_flip/map/while/stateful_uniform_full_int/ProdProd>random_flip/map/while/stateful_uniform_full_int/shape:output:0>random_flip/map/while/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: x
6random_flip/map/while/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :­
6random_flip/map/while/stateful_uniform_full_int/Cast_1Cast=random_flip/map/while/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Д
>random_flip/map/while/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipIrandom_flip_map_while_stateful_uniform_full_int_rngreadandskip_resource_0?random_flip/map/while/stateful_uniform_full_int/Cast/x:output:0:random_flip/map/while/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
Crandom_flip/map/while/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Erandom_flip/map/while/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Erandom_flip/map/while/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
=random_flip/map/while/stateful_uniform_full_int/strided_sliceStridedSliceFrandom_flip/map/while/stateful_uniform_full_int/RngReadAndSkip:value:0Lrandom_flip/map/while/stateful_uniform_full_int/strided_slice/stack:output:0Nrandom_flip/map/while/stateful_uniform_full_int/strided_slice/stack_1:output:0Nrandom_flip/map/while/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЛ
7random_flip/map/while/stateful_uniform_full_int/BitcastBitcastFrandom_flip/map/while/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Erandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Grandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Grandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
?random_flip/map/while/stateful_uniform_full_int/strided_slice_1StridedSliceFrandom_flip/map/while/stateful_uniform_full_int/RngReadAndSkip:value:0Nrandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stack:output:0Prandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Prandom_flip/map/while/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:П
9random_flip/map/while/stateful_uniform_full_int/Bitcast_1BitcastHrandom_flip/map/while/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0u
3random_flip/map/while/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :џ
/random_flip/map/while/stateful_uniform_full_intStatelessRandomUniformFullIntV2>random_flip/map/while/stateful_uniform_full_int/shape:output:0Brandom_flip/map/while/stateful_uniform_full_int/Bitcast_1:output:0@random_flip/map/while/stateful_uniform_full_int/Bitcast:output:0<random_flip/map/while/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	j
 random_flip/map/while/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R К
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
valueB"      х
#random_flip/map/while/strided_sliceStridedSlice$random_flip/map/while/stack:output:02random_flip/map/while/strided_slice/stack:output:04random_flip/map/while/strided_slice/stack_1:output:04random_flip/map/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
Irandom_flip/map/while/stateless_random_flip_left_right/control_dependencyIdentity@random_flip/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*L
_classB
@>loc:@random_flip/map/while/TensorArrayV2Read/TensorListGetItem*"
_output_shapes
:@@
Urandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Srandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Srandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?м
lrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter,random_flip/map/while/strided_slice:output:0* 
_output_shapes
::Ў
lrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :п
hrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2^random_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0rrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0vrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0urandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: З
Srandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/subSub\random_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/max:output:0\random_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Ч
Srandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/mulMulqrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Wrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: А
Orandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniformAddV2Wrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0\random_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
=random_flip/map/while/stateless_random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
;random_flip/map/while/stateless_random_flip_left_right/LessLessSrandom_flip/map/while/stateless_random_flip_left_right/stateless_random_uniform:z:0Frandom_flip/map/while/stateless_random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: Л
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
Arandom_flip_map_while_stateless_random_flip_left_right_true_11655Й
?random_flip/map/while/stateless_random_flip_left_right/IdentityIdentity?random_flip/map/while/stateless_random_flip_left_right:output:0*
T0*"
_output_shapes
:@@Ё
:random_flip/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#random_flip_map_while_placeholder_1!random_flip_map_while_placeholderHrandom_flip/map/while/stateless_random_flip_left_right/Identity:output:0*
_output_shapes
: *
element_dtype0:щшв]
random_flip/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
random_flip/map/while/addAddV2!random_flip_map_while_placeholder$random_flip/map/while/add/y:output:0*
T0*
_output_shapes
: _
random_flip/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ї
random_flip/map/while/add_1AddV28random_flip_map_while_random_flip_map_while_loop_counter&random_flip/map/while/add_1/y:output:0*
T0*
_output_shapes
: 
random_flip/map/while/IdentityIdentityrandom_flip/map/while/add_1:z:0^random_flip/map/while/NoOp*
T0*
_output_shapes
: 
 random_flip/map/while/Identity_1Identity3random_flip_map_while_random_flip_map_strided_slice^random_flip/map/while/NoOp*
T0*
_output_shapes
: 
 random_flip/map/while/Identity_2Identityrandom_flip/map/while/add:z:0^random_flip/map/while/NoOp*
T0*
_output_shapes
: Ж
 random_flip/map/while/Identity_3IdentityJrandom_flip/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^random_flip/map/while/NoOp*
T0*
_output_shapes
: 
random_flip/map/while/NoOpNoOp?^random_flip/map/while/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "I
random_flip_map_while_identity'random_flip/map/while/Identity:output:0"M
 random_flip_map_while_identity_1)random_flip/map/while/Identity_1:output:0"M
 random_flip_map_while_identity_2)random_flip/map/while/Identity_2:output:0"M
 random_flip_map_while_identity_3)random_flip/map/while/Identity_3:output:0"p
5random_flip_map_while_random_flip_map_strided_slice_17random_flip_map_while_random_flip_map_strided_slice_1_0"
Grandom_flip_map_while_stateful_uniform_full_int_rngreadandskip_resourceIrandom_flip_map_while_stateful_uniform_full_int_rngreadandskip_resource_0"ш
qrandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensorsrandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2
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
С
G
+__inference_random_flip_layer_call_fn_12005

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10013h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
І
я
E__inference_sequential_layer_call_and_return_conditional_losses_12348

inputs 
dense_3_12334:	@
dense_3_12336:	 
dense_4_12341:	@
dense_4_12343:@
identityЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallя
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_12334dense_3_12336*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_12162у
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12180ь
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12297
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_4_12341dense_4_12343*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_12219
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12264~
IdentityIdentity*dropout_2/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@в
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Л
H
,__inference_activation_1_layer_call_fn_13265

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_12461f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
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
:џџџџџџџџџP
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?m
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџY
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџO

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџe

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџ\
IdentityIdentityGelu/mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з
п
/__inference_swin_transformer_layer_call_fn_6442
x
unknown:@
	unknown_0:@
	unknown_1:	@Р
	unknown_2:	Р
	unknown_3:	
	unknown_4:	
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:	@

unknown_10:	

unknown_11:	@

unknown_12:@
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_swin_transformer_layer_call_and_return_conditional_losses_6423`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
њ(
ъ
I__inference_patch_embedding_layer_call_and_return_conditional_losses_1264	
patch9
'dense_tensordot_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@4
!embedding_embedding_lookup_368004:	@
identityЂdense/BiasAdd/ReadVariableOpЂdense/Tensordot/ReadVariableOpЂembedding/embedding_lookupM
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
B :M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :m
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:
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
value	B : г
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
value	B : з
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
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposepatchdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : П
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@г
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_368004range:output:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/368004*
_output_shapes
:	@*
dtype0Д
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/368004*
_output_shapes
:	@
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	@
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ѓ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 [
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:S O
,
_output_shapes
:џџџџџџџџџ

_user_specified_namepatch
Ш
й
,__inference_sequential_1_layer_call_fn_12525
dense_7_input
unknown:	@
	unknown_0:	
	unknown_1:	@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_12514t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ@
'
_user_specified_namedense_7_input
Й&
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
§џџџџџџџџm
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџd
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
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
ўџџџџџџџџo
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџf
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ђ
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
 *  B`
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
 *  Bf
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
value	B : 

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
џџџџџџџџџY
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:Ђ

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"џџџџџџџџџ@@џџџџџџџџџa
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   А
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
half_pixel_centers(
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:џџџџџџџџџ@@"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: :5 1
/
_output_shapes
:џџџџџџџџџ@@
V
г
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
;map_while_stateful_uniform_full_int_rngreadandskip_resource:	Ђ2map/while/stateful_uniform_full_int/RngReadAndSkip
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      Е
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
valueB: Й
(map/while/stateful_uniform_full_int/ProdProd2map/while/stateful_uniform_full_int/shape:output:02map/while/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: l
*map/while/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
*map/while/stateful_uniform_full_int/Cast_1Cast1map/while/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
2map/while/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=map_while_stateful_uniform_full_int_rngreadandskip_resource_03map/while/stateful_uniform_full_int/Cast/x:output:0.map/while/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
7map/while/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9map/while/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9map/while/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1map/while/stateful_uniform_full_int/strided_sliceStridedSlice:map/while/stateful_uniform_full_int/RngReadAndSkip:value:0@map/while/stateful_uniform_full_int/strided_slice/stack:output:0Bmap/while/stateful_uniform_full_int/strided_slice/stack_1:output:0Bmap/while/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЃ
+map/while/stateful_uniform_full_int/BitcastBitcast:map/while/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
9map/while/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;map/while/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;map/while/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3map/while/stateful_uniform_full_int/strided_slice_1StridedSlice:map/while/stateful_uniform_full_int/RngReadAndSkip:value:0Bmap/while/stateful_uniform_full_int/strided_slice_1/stack:output:0Dmap/while/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Dmap/while/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ї
-map/while/stateful_uniform_full_int/Bitcast_1Bitcast<map/while/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'map/while/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :У
#map/while/stateful_uniform_full_intStatelessRandomUniformFullIntV22map/while/stateful_uniform_full_int/shape:output:06map/while/stateful_uniform_full_int/Bitcast_1:output:04map/while/stateful_uniform_full_int/Bitcast:output:00map/while/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	^
map/while/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 
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
valueB"      Љ
map/while/strided_sliceStridedSlicemap/while/stack:output:0&map/while/strided_slice/stack:output:0(map/while/strided_slice/stack_1:output:0(map/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskю
=map/while/stateless_random_flip_left_right/control_dependencyIdentity4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*@
_class6
42loc:@map/while/TensorArrayV2Read/TensorListGetItem*"
_output_shapes
:@@
Imap/while/stateless_random_flip_left_right/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ф
`map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter map/while/strided_slice:output:0* 
_output_shapes
::Ђ
`map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
\map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Rmap/while/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0fmap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0jmap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0imap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: 
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/subSubPmap/while/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Pmap/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Ѓ
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/mulMulemap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Kmap/while/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 
Cmap/while/stateless_random_flip_left_right/stateless_random_uniformAddV2Kmap/while/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Pmap/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: v
1map/while/stateless_random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?н
/map/while/stateless_random_flip_left_right/LessLessGmap/while/stateless_random_flip_left_right/stateless_random_uniform:z:0:map/while/stateless_random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: џ
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
5map_while_stateless_random_flip_left_right_true_10271Ё
3map/while/stateless_random_flip_left_right/IdentityIdentity3map/while/stateless_random_flip_left_right:output:0*
T0*"
_output_shapes
:@@ё
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder<map/while/stateless_random_flip_left_right/Identity:output:0*
_output_shapes
: *
element_dtype0:щшвQ
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
: 
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: 
map/while/NoOpNoOp3^map/while/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"|
;map_while_stateful_uniform_full_int_rngreadandskip_resource=map_while_stateful_uniform_full_int_rngreadandskip_resource_0"И
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
я
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_12187

inputs

identity_1T
IdentityIdentityinputs*
T0*-
_output_shapes
:џџџџџџџџџa

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ9
Ы
@__inference_model_layer_call_and_return_conditional_losses_10728

inputs
random_crop_10576:	
random_flip_10579:	'
patch_embedding_10583:@#
patch_embedding_10585:@(
patch_embedding_10587:	@$
swin_transformer_10623:@$
swin_transformer_10625:@)
swin_transformer_10627:	@Р%
swin_transformer_10629:	Р(
swin_transformer_10631:	(
swin_transformer_10633:	(
swin_transformer_10635:@@$
swin_transformer_10637:@$
swin_transformer_10639:@$
swin_transformer_10641:@)
swin_transformer_10643:	@%
swin_transformer_10645:	)
swin_transformer_10647:	@$
swin_transformer_10649:@&
swin_transformer_1_10687:@&
swin_transformer_1_10689:@+
swin_transformer_1_10691:	@Р'
swin_transformer_1_10693:	Р*
swin_transformer_1_10695:	*
swin_transformer_1_10697:	/
swin_transformer_1_10699:*
swin_transformer_1_10701:@@&
swin_transformer_1_10703:@&
swin_transformer_1_10705:@&
swin_transformer_1_10707:@+
swin_transformer_1_10709:	@'
swin_transformer_1_10711:	+
swin_transformer_1_10713:	@&
swin_transformer_1_10715:@'
patch_merging_10718:
!
dense_10_10722:	W
dense_10_10724:W
identityЂ dense_10/StatefulPartitionedCallЂ'patch_embedding/StatefulPartitionedCallЂ%patch_merging/StatefulPartitionedCallЂ#random_crop/StatefulPartitionedCallЂ#random_flip/StatefulPartitionedCallЂ(swin_transformer/StatefulPartitionedCallЂ*swin_transformer_1/StatefulPartitionedCallщ
#random_crop/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_crop_10576*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10491
#random_flip/StatefulPartitionedCallStatefulPartitionedCall,random_crop/StatefulPartitionedCall:output:0random_flip_10579*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10305Ы
patch_extract/PartitionedCallPartitionedCall,random_flip/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9773Є
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_10583patch_embedding_10585patch_embedding_10587*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9785б
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_10623swin_transformer_10625swin_transformer_10627swin_transformer_10629swin_transformer_10631swin_transformer_10633swin_transformer_10635swin_transformer_10637swin_transformer_10639swin_transformer_10641swin_transformer_10643swin_transformer_10645swin_transformer_10647swin_transformer_10649*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_restored_function_body_10622
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_10687swin_transformer_1_10689swin_transformer_1_10691swin_transformer_1_10693swin_transformer_1_10695swin_transformer_1_10697swin_transformer_1_10699swin_transformer_1_10701swin_transformer_1_10703swin_transformer_1_10705swin_transformer_1_10707swin_transformer_1_10709swin_transformer_1_10711swin_transformer_1_10713swin_transformer_1_10715*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_restored_function_body_10686ќ
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_10718*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9927џ
(global_average_pooling1d/PartitionedCallPartitionedCall.patch_merging/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951
 dense_10/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_10_10722dense_10_10724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџW*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_10098x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџWп
NoOpNoOp!^dense_10/StatefulPartitionedCall(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall$^random_crop/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2J
#random_crop/StatefulPartitionedCall#random_crop/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ы
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_13358

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџ@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
е

'__inference_dense_4_layer_call_fn_13164

inputs
unknown:	@
	unknown_0:@
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_12219t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
д
њ
B__inference_dense_4_layer_call_and_return_conditional_losses_12219

inputs4
!tensordot_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


c
D__inference_dropout_5_layer_call_and_return_conditional_losses_12545

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Ћ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
и
ћ
B__inference_dense_3_layer_call_and_return_conditional_losses_13111

inputs4
!tensordot_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџe
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ђ

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_12578

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?j
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Ќ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:џџџџџџџџџu
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:џџџџџџџџџo
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:џџџџџџџџџ_
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р
њ
'__inference_restored_function_body_9889
x
unknown:@
	unknown_0:@
	unknown_1:	@Р
	unknown_2:	Р
	unknown_3:	
	unknown_4:	 
	unknown_5:
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:	@

unknown_11:	

unknown_12:	@

unknown_13:@
identityЂStatefulPartitionedCallя
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
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_2354t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ@: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
Ш
й
,__inference_sequential_1_layer_call_fn_12653
dense_7_input
unknown:	@
	unknown_0:	
	unknown_1:	@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_12629t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ@
'
_user_specified_namedense_7_input
V
г
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
;map_while_stateful_uniform_full_int_rngreadandskip_resource:	Ђ2map/while/stateful_uniform_full_int/RngReadAndSkip
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      Е
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
valueB: Й
(map/while/stateful_uniform_full_int/ProdProd2map/while/stateful_uniform_full_int/shape:output:02map/while/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: l
*map/while/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
*map/while/stateful_uniform_full_int/Cast_1Cast1map/while/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
2map/while/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=map_while_stateful_uniform_full_int_rngreadandskip_resource_03map/while/stateful_uniform_full_int/Cast/x:output:0.map/while/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
7map/while/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9map/while/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9map/while/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1map/while/stateful_uniform_full_int/strided_sliceStridedSlice:map/while/stateful_uniform_full_int/RngReadAndSkip:value:0@map/while/stateful_uniform_full_int/strided_slice/stack:output:0Bmap/while/stateful_uniform_full_int/strided_slice/stack_1:output:0Bmap/while/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЃ
+map/while/stateful_uniform_full_int/BitcastBitcast:map/while/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
9map/while/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;map/while/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;map/while/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3map/while/stateful_uniform_full_int/strided_slice_1StridedSlice:map/while/stateful_uniform_full_int/RngReadAndSkip:value:0Bmap/while/stateful_uniform_full_int/strided_slice_1/stack:output:0Dmap/while/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Dmap/while/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ї
-map/while/stateful_uniform_full_int/Bitcast_1Bitcast<map/while/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'map/while/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :У
#map/while/stateful_uniform_full_intStatelessRandomUniformFullIntV22map/while/stateful_uniform_full_int/shape:output:06map/while/stateful_uniform_full_int/Bitcast_1:output:04map/while/stateful_uniform_full_int/Bitcast:output:00map/while/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	^
map/while/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 
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
valueB"      Љ
map/while/strided_sliceStridedSlicemap/while/stack:output:0&map/while/strided_slice/stack:output:0(map/while/strided_slice/stack_1:output:0(map/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskю
=map/while/stateless_random_flip_left_right/control_dependencyIdentity4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*@
_class6
42loc:@map/while/TensorArrayV2Read/TensorListGetItem*"
_output_shapes
:@@
Imap/while/stateless_random_flip_left_right/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ф
`map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter map/while/strided_slice:output:0* 
_output_shapes
::Ђ
`map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
\map/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Rmap/while/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0fmap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0jmap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0imap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: 
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/subSubPmap/while/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Pmap/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Ѓ
Gmap/while/stateless_random_flip_left_right/stateless_random_uniform/mulMulemap/while/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Kmap/while/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 
Cmap/while/stateless_random_flip_left_right/stateless_random_uniformAddV2Kmap/while/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Pmap/while/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: v
1map/while/stateless_random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?н
/map/while/stateless_random_flip_left_right/LessLessGmap/while/stateless_random_flip_left_right/stateless_random_uniform:z:0:map/while/stateless_random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: џ
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
5map_while_stateless_random_flip_left_right_true_12091Ё
3map/while/stateless_random_flip_left_right/IdentityIdentity3map/while/stateless_random_flip_left_right:output:0*
T0*"
_output_shapes
:@@ё
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder<map/while/stateless_random_flip_left_right/Identity:output:0*
_output_shapes
: *
element_dtype0:щшвQ
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
: 
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: 
map/while/NoOpNoOp3^map/while/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"|
;map_while_stateful_uniform_full_int_rngreadandskip_resource=map_while_stateful_uniform_full_int_rngreadandskip_resource_0"И
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
	
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
:џџџџџџџџџP
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?m
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџY
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџO

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџe

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџ\
IdentityIdentityGelu/mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Њ
з
'__inference_restored_function_body_9825
x
unknown:@
	unknown_0:@
	unknown_1:	@Р
	unknown_2:	Р
	unknown_3:	
	unknown_4:	
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:	@

unknown_10:	

unknown_11:	@

unknown_12:@
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*,
_output_shapes
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_swin_transformer_layer_call_and_return_conditional_losses_721t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
У
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
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
ExtractImagePatchesExtractImagePatchesimages*
T0*/
_output_shapes
:џџџџџџџџџ  *
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
B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameimages


F__inference_random_flip_layer_call_and_return_conditional_losses_12125

inputs
map_while_input_6:	
identityЂ	map/while?
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
valueB:х
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
џџџџџџџџџО
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      с
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputsBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвK
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
џџџџџџџџџТ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвX
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
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
: : : : : : : 
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      в
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*/
_output_shapes
:џџџџџџџџџ@@*
element_dtype0
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@R
NoOpNoOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: 2
	map/while	map/while:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Џ
а
*__inference_sequential_layer_call_fn_12754

inputs
unknown:	@
	unknown_0:	
	unknown_1:	@
	unknown_2:@
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12348t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
У
ј
G__inference_sequential_1_layer_call_and_return_conditional_losses_12687
dense_7_input 
dense_7_12673:	@
dense_7_12675:	 
dense_8_12680:	@
dense_8_12682:@
identityЂdense_7/StatefulPartitionedCallЂdense_8/StatefulPartitionedCallЂ!dropout_4/StatefulPartitionedCallЂ!dropout_5/StatefulPartitionedCallі
dense_7/StatefulPartitionedCallStatefulPartitionedCalldense_7_inputdense_7_12673dense_7_12675*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_12443ч
activation_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_12461ю
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12578
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_8_12680dense_8_12682*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_12500
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12545~
IdentityIdentity*dropout_5/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@в
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ@
'
_user_specified_namedense_7_input
Ш

F__inference_random_crop_layer_call_and_return_conditional_losses_10491

inputs
cond_input_1:	
identityЂcond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџh
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
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
ўџџџџџџџџj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
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
: Ц
condIfAll:output:0inputscond_input_1*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_10346*.
output_shapes
:џџџџџџџџџ@@*"
then_branchR
cond_true_10345b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: 2
condcond:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
єж

L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_1156
xI
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@O
<window_attention_1_dense_5_tensordot_readvariableop_resource:	@РI
:window_attention_1_dense_5_biasadd_readvariableop_resource:	РF
4window_attention_1_reshape_1_readvariableop_resource:	4
"window_attention_1_gather_resource:	N
7window_attention_1_expanddims_1_readvariableop_resource:N
<window_attention_1_dense_6_tensordot_readvariableop_resource:@@H
:window_attention_1_dense_6_biasadd_readvariableop_resource:@I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_3_batchnorm_readvariableop_resource:@I
6sequential_1_dense_7_tensordot_readvariableop_resource:	@C
4sequential_1_dense_7_biasadd_readvariableop_resource:	I
6sequential_1_dense_8_tensordot_readvariableop_resource:	@B
4sequential_1_dense_8_biasadd_readvariableop_resource:@
identityЂ.layer_normalization_2/batchnorm/ReadVariableOpЂ2layer_normalization_2/batchnorm/mul/ReadVariableOpЂ.layer_normalization_3/batchnorm/ReadVariableOpЂ2layer_normalization_3/batchnorm/mul/ReadVariableOpЂ+sequential_1/dense_7/BiasAdd/ReadVariableOpЂ-sequential_1/dense_7/Tensordot/ReadVariableOpЂ+sequential_1/dense_8/BiasAdd/ReadVariableOpЂ-sequential_1/dense_8/Tensordot/ReadVariableOpЂ.window_attention_1/ExpandDims_1/ReadVariableOpЂwindow_attention_1/GatherЂ+window_attention_1/Reshape_1/ReadVariableOpЂ1window_attention_1/dense_5/BiasAdd/ReadVariableOpЂ3window_attention_1/dense_5/Tensordot/ReadVariableOpЂ1window_attention_1/dense_6/BiasAdd/ReadVariableOpЂ3window_attention_1/dense_6/Tensordot/ReadVariableOp~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Д
"layer_normalization_2/moments/meanMeanx=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencex3layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_2/batchnorm/mul_1Mulx'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   
ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @[

Roll/shiftConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџZ
	Roll/axisConst*
_output_shapes
:*
dtype0*
valueB"      
RollRollReshape:output:0Roll/shift:output:0Roll/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:џџџџџџџџџ  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_1ReshapeRoll:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Б
3window_attention_1/dense_5/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@Р*
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
value	B : Ї
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
value	B : Ћ
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
valueB: П
)window_attention_1/dense_5/Tensordot/ProdProd6window_attention_1/dense_5/Tensordot/GatherV2:output:03window_attention_1/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Х
+window_attention_1/dense_5/Tensordot/Prod_1Prod8window_attention_1/dense_5/Tensordot/GatherV2_1:output:05window_attention_1/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention_1/dense_5/Tensordot/concatConcatV22window_attention_1/dense_5/Tensordot/free:output:02window_attention_1/dense_5/Tensordot/axes:output:09window_attention_1/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ъ
*window_attention_1/dense_5/Tensordot/stackPack2window_attention_1/dense_5/Tensordot/Prod:output:04window_attention_1/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Л
.window_attention_1/dense_5/Tensordot/transpose	TransposeReshape_3:output:04window_attention_1/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@л
,window_attention_1/dense_5/Tensordot/ReshapeReshape2window_attention_1/dense_5/Tensordot/transpose:y:03window_attention_1/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџм
+window_attention_1/dense_5/Tensordot/MatMulMatMul5window_attention_1/dense_5/Tensordot/Reshape:output:0;window_attention_1/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџРw
,window_attention_1/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Рt
2window_attention_1/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-window_attention_1/dense_5/Tensordot/concat_1ConcatV26window_attention_1/dense_5/Tensordot/GatherV2:output:05window_attention_1/dense_5/Tensordot/Const_2:output:0;window_attention_1/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:е
$window_attention_1/dense_5/TensordotReshape5window_attention_1/dense_5/Tensordot/MatMul:product:06window_attention_1/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџРЉ
1window_attention_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype0Ю
"window_attention_1/dense_5/BiasAddBiasAdd-window_attention_1/dense_5/Tensordot:output:09window_attention_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџР}
 window_attention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Л
window_attention_1/ReshapeReshape+window_attention_1/dense_5/BiasAdd:output:0)window_attention_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ~
!window_attention_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                И
window_attention_1/transpose	Transpose#window_attention_1/Reshape:output:0*window_attention_1/transpose/perm:output:0*
T0*3
_output_shapes!
:џџџџџџџџџp
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
valueB:Ш
 window_attention_1/strided_sliceStridedSlice window_attention_1/transpose:y:0/window_attention_1/strided_slice/stack:output:01window_attention_1/strided_slice/stack_1:output:01window_attention_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:а
"window_attention_1/strided_slice_1StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_1/stack:output:03window_attention_1/strided_slice_1/stack_1:output:03window_attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:а
"window_attention_1/strided_slice_2StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_2/stack:output:03window_attention_1/strided_slice_2/stack_1:output:03window_attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask]
window_attention_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ>Ѕ
window_attention_1/mulMul)window_attention_1/strided_slice:output:0!window_attention_1/mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|
#window_attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Р
window_attention_1/transpose_1	Transpose+window_attention_1/strided_slice_1:output:0,window_attention_1/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџЄ
window_attention_1/matmulBatchMatMulV2window_attention_1/mul:z:0"window_attention_1/transpose_1:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 
+window_attention_1/Reshape_1/ReadVariableOpReadVariableOp4window_attention_1_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	u
"window_attention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЎ
window_attention_1/Reshape_1Reshape3window_attention_1/Reshape_1/ReadVariableOp:value:0+window_attention_1/Reshape_1/shape:output:0*
T0	*
_output_shapes
:Г
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
valueB"      џџџџЇ
window_attention_1/Reshape_2Reshape$window_attention_1/Identity:output:0+window_attention_1/Reshape_2/shape:output:0*
T0*"
_output_shapes
:x
#window_attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
window_attention_1/transpose_2	Transpose%window_attention_1/Reshape_2:output:0,window_attention_1/transpose_2/perm:output:0*
T0*"
_output_shapes
:c
!window_attention_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ќ
window_attention_1/ExpandDims
ExpandDims"window_attention_1/transpose_2:y:0*window_attention_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:Ѕ
window_attention_1/addAddV2"window_attention_1/matmul:output:0&window_attention_1/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџЋ
.window_attention_1/ExpandDims_1/ReadVariableOpReadVariableOp7window_attention_1_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype0e
#window_attention_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :Х
window_attention_1/ExpandDims_1
ExpandDims6window_attention_1/ExpandDims_1/ReadVariableOp:value:0,window_attention_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:e
#window_attention_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : Л
window_attention_1/ExpandDims_2
ExpandDims(window_attention_1/ExpandDims_1:output:0,window_attention_1/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:
window_attention_1/CastCast(window_attention_1/ExpandDims_2:output:0*

DstT0*

SrcT0*+
_output_shapes
:
"window_attention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Џ
window_attention_1/Reshape_3Reshapewindow_attention_1/add:z:0+window_attention_1/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџЄ
window_attention_1/add_1AddV2%window_attention_1/Reshape_3:output:0window_attention_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџ{
"window_attention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         Ќ
window_attention_1/Reshape_4Reshapewindow_attention_1/add_1:z:0+window_attention_1/Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
window_attention_1/SoftmaxSoftmax%window_attention_1/Reshape_4:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
%window_attention_1/dropout_3/IdentityIdentity$window_attention_1/Softmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџУ
window_attention_1/matmul_1BatchMatMulV2.window_attention_1/dropout_3/Identity:output:0+window_attention_1/strided_slice_2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|
#window_attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Й
window_attention_1/transpose_3	Transpose$window_attention_1/matmul_1:output:0,window_attention_1/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
"window_attention_1/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   Ў
window_attention_1/Reshape_5Reshape"window_attention_1/transpose_3:y:0+window_attention_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@А
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
value	B : Ї
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
value	B : Ћ
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
valueB: П
)window_attention_1/dense_6/Tensordot/ProdProd6window_attention_1/dense_6/Tensordot/GatherV2:output:03window_attention_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Х
+window_attention_1/dense_6/Tensordot/Prod_1Prod8window_attention_1/dense_6/Tensordot/GatherV2_1:output:05window_attention_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention_1/dense_6/Tensordot/concatConcatV22window_attention_1/dense_6/Tensordot/free:output:02window_attention_1/dense_6/Tensordot/axes:output:09window_attention_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ъ
*window_attention_1/dense_6/Tensordot/stackPack2window_attention_1/dense_6/Tensordot/Prod:output:04window_attention_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ю
.window_attention_1/dense_6/Tensordot/transpose	Transpose%window_attention_1/Reshape_5:output:04window_attention_1/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@л
,window_attention_1/dense_6/Tensordot/ReshapeReshape2window_attention_1/dense_6/Tensordot/transpose:y:03window_attention_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџл
+window_attention_1/dense_6/Tensordot/MatMulMatMul5window_attention_1/dense_6/Tensordot/Reshape:output:0;window_attention_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
,window_attention_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2window_attention_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-window_attention_1/dense_6/Tensordot/concat_1ConcatV26window_attention_1/dense_6/Tensordot/GatherV2:output:05window_attention_1/dense_6/Tensordot/Const_2:output:0;window_attention_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:д
$window_attention_1/dense_6/TensordotReshape5window_attention_1/dense_6/Tensordot/MatMul:product:06window_attention_1/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ј
1window_attention_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Э
"window_attention_1/dense_6/BiasAddBiasAdd-window_attention_1/dense_6/Tensordot:output:09window_attention_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@
'window_attention_1/dropout_3/Identity_1Identity+window_attention_1/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   
	Reshape_4Reshape0window_attention_1/dropout_3/Identity_1:output:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @]
Roll_1/shiftConst*
_output_shapes
:*
dtype0*
valueB"      \
Roll_1/axisConst*
_output_shapes
:*
dtype0*
valueB"      Є
Roll_1RollReshape_6:output:0Roll_1/shift:output:0Roll_1/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:џџџџџџџџџ  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   v
	Reshape_7ReshapeRoll_1:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@S
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
valueB:
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
value	B :д
 drop_path_1/random_uniform/shapePack"drop_path_1/strided_slice:output:0+drop_path_1/random_uniform/shape/1:output:0+drop_path_1/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:Ї
(drop_path_1/random_uniform/RandomUniformRandomUniform)drop_path_1/random_uniform/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0V
drop_path_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path_1/addAddV2drop_path_1/add/x:output:01drop_path_1/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџe
drop_path_1/FloorFloordrop_path_1/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
drop_path_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path_1/truedivRealDivReshape_7:output:0drop_path_1/truediv/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@}
drop_path_1/mulMuldrop_path_1/truediv:z:0drop_path_1/Floor:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@[
addAddV2xdrop_path_1/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:К
"layer_normalization_3/moments/meanMeanadd:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЙ
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_3/batchnorm/mul_1Muladd:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ѕ
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: ­
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Г
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : №
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:И
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ч
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Щ
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЪ
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџq
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ћ
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ф
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Н
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџi
$sequential_1/activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?З
"sequential_1/activation_1/Gelu/mulMul-sequential_1/activation_1/Gelu/mul/x:output:0%sequential_1/dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџj
%sequential_1/activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?Р
&sequential_1/activation_1/Gelu/truedivRealDiv%sequential_1/dense_7/BiasAdd:output:0.sequential_1/activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
"sequential_1/activation_1/Gelu/ErfErf*sequential_1/activation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџi
$sequential_1/activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?К
"sequential_1/activation_1/Gelu/addAddV2-sequential_1/activation_1/Gelu/add/x:output:0&sequential_1/activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџГ
$sequential_1/activation_1/Gelu/mul_1Mul&sequential_1/activation_1/Gelu/mul:z:0&sequential_1/activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџ
sequential_1/dropout_4/IdentityIdentity(sequential_1/activation_1/Gelu/mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџЅ
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: ­
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Г
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : №
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:И
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ч
(sequential_1/dense_8/Tensordot/transpose	Transpose(sequential_1/dropout_4/Identity:output:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџЩ
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЩ
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ћ
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:У
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0М
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
sequential_1/dropout_5/IdentityIdentity%sequential_1/dense_8/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@k
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
valueB:
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
value	B :м
"drop_path_1/random_uniform_1/shapePack$drop_path_1/strided_slice_1:output:0-drop_path_1/random_uniform_1/shape/1:output:0-drop_path_1/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ћ
*drop_path_1/random_uniform_1/RandomUniformRandomUniform+drop_path_1/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0X
drop_path_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?Ѓ
drop_path_1/add_1AddV2drop_path_1/add_1/x:output:03drop_path_1/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџi
drop_path_1/Floor_1Floordrop_path_1/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ\
drop_path_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?Ѓ
drop_path_1/truediv_1RealDiv(sequential_1/dropout_5/Identity:output:0 drop_path_1/truediv_1/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
drop_path_1/mul_1Muldrop_path_1/truediv_1:z:0drop_path_1/Floor_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@e
add_1AddV2add:z:0drop_path_1/mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@
NoOpNoOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp/^window_attention_1/ExpandDims_1/ReadVariableOp^window_attention_1/Gather,^window_attention_1/Reshape_1/ReadVariableOp2^window_attention_1/dense_5/BiasAdd/ReadVariableOp4^window_attention_1/dense_5/Tensordot/ReadVariableOp2^window_attention_1/dense_6/BiasAdd/ReadVariableOp4^window_attention_1/dense_6/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ@: : : : : : : : : : : : : : : 2`
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
:џџџџџџџџџ@

_user_specified_namex
Љ§

L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_1603
xI
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@O
<window_attention_1_dense_5_tensordot_readvariableop_resource:	@РI
:window_attention_1_dense_5_biasadd_readvariableop_resource:	РF
4window_attention_1_reshape_1_readvariableop_resource:	4
"window_attention_1_gather_resource:	N
7window_attention_1_expanddims_1_readvariableop_resource:N
<window_attention_1_dense_6_tensordot_readvariableop_resource:@@H
:window_attention_1_dense_6_biasadd_readvariableop_resource:@I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_3_batchnorm_readvariableop_resource:@I
6sequential_1_dense_7_tensordot_readvariableop_resource:	@C
4sequential_1_dense_7_biasadd_readvariableop_resource:	I
6sequential_1_dense_8_tensordot_readvariableop_resource:	@B
4sequential_1_dense_8_biasadd_readvariableop_resource:@
identityЂ.layer_normalization_2/batchnorm/ReadVariableOpЂ2layer_normalization_2/batchnorm/mul/ReadVariableOpЂ.layer_normalization_3/batchnorm/ReadVariableOpЂ2layer_normalization_3/batchnorm/mul/ReadVariableOpЂ+sequential_1/dense_7/BiasAdd/ReadVariableOpЂ-sequential_1/dense_7/Tensordot/ReadVariableOpЂ+sequential_1/dense_8/BiasAdd/ReadVariableOpЂ-sequential_1/dense_8/Tensordot/ReadVariableOpЂ.window_attention_1/ExpandDims_1/ReadVariableOpЂwindow_attention_1/GatherЂ+window_attention_1/Reshape_1/ReadVariableOpЂ1window_attention_1/dense_5/BiasAdd/ReadVariableOpЂ3window_attention_1/dense_5/Tensordot/ReadVariableOpЂ1window_attention_1/dense_6/BiasAdd/ReadVariableOpЂ3window_attention_1/dense_6/Tensordot/ReadVariableOp~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Д
"layer_normalization_2/moments/meanMeanx=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencex3layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_2/batchnorm/mul_1Mulx'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   
ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @[

Roll/shiftConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџZ
	Roll/axisConst*
_output_shapes
:*
dtype0*
valueB"      
RollRollReshape:output:0Roll/shift:output:0Roll/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:џџџџџџџџџ  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_1ReshapeRoll:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Б
3window_attention_1/dense_5/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@Р*
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
value	B : Ї
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
value	B : Ћ
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
valueB: П
)window_attention_1/dense_5/Tensordot/ProdProd6window_attention_1/dense_5/Tensordot/GatherV2:output:03window_attention_1/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Х
+window_attention_1/dense_5/Tensordot/Prod_1Prod8window_attention_1/dense_5/Tensordot/GatherV2_1:output:05window_attention_1/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention_1/dense_5/Tensordot/concatConcatV22window_attention_1/dense_5/Tensordot/free:output:02window_attention_1/dense_5/Tensordot/axes:output:09window_attention_1/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ъ
*window_attention_1/dense_5/Tensordot/stackPack2window_attention_1/dense_5/Tensordot/Prod:output:04window_attention_1/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Л
.window_attention_1/dense_5/Tensordot/transpose	TransposeReshape_3:output:04window_attention_1/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@л
,window_attention_1/dense_5/Tensordot/ReshapeReshape2window_attention_1/dense_5/Tensordot/transpose:y:03window_attention_1/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџм
+window_attention_1/dense_5/Tensordot/MatMulMatMul5window_attention_1/dense_5/Tensordot/Reshape:output:0;window_attention_1/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџРw
,window_attention_1/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Рt
2window_attention_1/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-window_attention_1/dense_5/Tensordot/concat_1ConcatV26window_attention_1/dense_5/Tensordot/GatherV2:output:05window_attention_1/dense_5/Tensordot/Const_2:output:0;window_attention_1/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:е
$window_attention_1/dense_5/TensordotReshape5window_attention_1/dense_5/Tensordot/MatMul:product:06window_attention_1/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџРЉ
1window_attention_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype0Ю
"window_attention_1/dense_5/BiasAddBiasAdd-window_attention_1/dense_5/Tensordot:output:09window_attention_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџР}
 window_attention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Л
window_attention_1/ReshapeReshape+window_attention_1/dense_5/BiasAdd:output:0)window_attention_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ~
!window_attention_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                И
window_attention_1/transpose	Transpose#window_attention_1/Reshape:output:0*window_attention_1/transpose/perm:output:0*
T0*3
_output_shapes!
:џџџџџџџџџp
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
valueB:Ш
 window_attention_1/strided_sliceStridedSlice window_attention_1/transpose:y:0/window_attention_1/strided_slice/stack:output:01window_attention_1/strided_slice/stack_1:output:01window_attention_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:а
"window_attention_1/strided_slice_1StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_1/stack:output:03window_attention_1/strided_slice_1/stack_1:output:03window_attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:а
"window_attention_1/strided_slice_2StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_2/stack:output:03window_attention_1/strided_slice_2/stack_1:output:03window_attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask]
window_attention_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ>Ѕ
window_attention_1/mulMul)window_attention_1/strided_slice:output:0!window_attention_1/mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|
#window_attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Р
window_attention_1/transpose_1	Transpose+window_attention_1/strided_slice_1:output:0,window_attention_1/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџЄ
window_attention_1/matmulBatchMatMulV2window_attention_1/mul:z:0"window_attention_1/transpose_1:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 
+window_attention_1/Reshape_1/ReadVariableOpReadVariableOp4window_attention_1_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	u
"window_attention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЎ
window_attention_1/Reshape_1Reshape3window_attention_1/Reshape_1/ReadVariableOp:value:0+window_attention_1/Reshape_1/shape:output:0*
T0	*
_output_shapes
:Г
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
valueB"      џџџџЇ
window_attention_1/Reshape_2Reshape$window_attention_1/Identity:output:0+window_attention_1/Reshape_2/shape:output:0*
T0*"
_output_shapes
:x
#window_attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
window_attention_1/transpose_2	Transpose%window_attention_1/Reshape_2:output:0,window_attention_1/transpose_2/perm:output:0*
T0*"
_output_shapes
:c
!window_attention_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ќ
window_attention_1/ExpandDims
ExpandDims"window_attention_1/transpose_2:y:0*window_attention_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:Ѕ
window_attention_1/addAddV2"window_attention_1/matmul:output:0&window_attention_1/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџЋ
.window_attention_1/ExpandDims_1/ReadVariableOpReadVariableOp7window_attention_1_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype0e
#window_attention_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :Х
window_attention_1/ExpandDims_1
ExpandDims6window_attention_1/ExpandDims_1/ReadVariableOp:value:0,window_attention_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:e
#window_attention_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : Л
window_attention_1/ExpandDims_2
ExpandDims(window_attention_1/ExpandDims_1:output:0,window_attention_1/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:
window_attention_1/CastCast(window_attention_1/ExpandDims_2:output:0*

DstT0*

SrcT0*+
_output_shapes
:
"window_attention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Џ
window_attention_1/Reshape_3Reshapewindow_attention_1/add:z:0+window_attention_1/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџЄ
window_attention_1/add_1AddV2%window_attention_1/Reshape_3:output:0window_attention_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџ{
"window_attention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         Ќ
window_attention_1/Reshape_4Reshapewindow_attention_1/add_1:z:0+window_attention_1/Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
window_attention_1/SoftmaxSoftmax%window_attention_1/Reshape_4:output:0*
T0*/
_output_shapes
:џџџџџџџџџo
*window_attention_1/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?Ф
(window_attention_1/dropout_3/dropout/MulMul$window_attention_1/Softmax:softmax:03window_attention_1/dropout_3/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ~
*window_attention_1/dropout_3/dropout/ShapeShape$window_attention_1/Softmax:softmax:0*
T0*
_output_shapes
:Ю
Awindow_attention_1/dropout_3/dropout/random_uniform/RandomUniformRandomUniform3window_attention_1/dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0x
3window_attention_1/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<
1window_attention_1/dropout_3/dropout/GreaterEqualGreaterEqualJwindow_attention_1/dropout_3/dropout/random_uniform/RandomUniform:output:0<window_attention_1/dropout_3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџБ
)window_attention_1/dropout_3/dropout/CastCast5window_attention_1/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџШ
*window_attention_1/dropout_3/dropout/Mul_1Mul,window_attention_1/dropout_3/dropout/Mul:z:0-window_attention_1/dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџУ
window_attention_1/matmul_1BatchMatMulV2.window_attention_1/dropout_3/dropout/Mul_1:z:0+window_attention_1/strided_slice_2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|
#window_attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Й
window_attention_1/transpose_3	Transpose$window_attention_1/matmul_1:output:0,window_attention_1/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
"window_attention_1/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   Ў
window_attention_1/Reshape_5Reshape"window_attention_1/transpose_3:y:0+window_attention_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@А
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
value	B : Ї
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
value	B : Ћ
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
valueB: П
)window_attention_1/dense_6/Tensordot/ProdProd6window_attention_1/dense_6/Tensordot/GatherV2:output:03window_attention_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Х
+window_attention_1/dense_6/Tensordot/Prod_1Prod8window_attention_1/dense_6/Tensordot/GatherV2_1:output:05window_attention_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention_1/dense_6/Tensordot/concatConcatV22window_attention_1/dense_6/Tensordot/free:output:02window_attention_1/dense_6/Tensordot/axes:output:09window_attention_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ъ
*window_attention_1/dense_6/Tensordot/stackPack2window_attention_1/dense_6/Tensordot/Prod:output:04window_attention_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ю
.window_attention_1/dense_6/Tensordot/transpose	Transpose%window_attention_1/Reshape_5:output:04window_attention_1/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@л
,window_attention_1/dense_6/Tensordot/ReshapeReshape2window_attention_1/dense_6/Tensordot/transpose:y:03window_attention_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџл
+window_attention_1/dense_6/Tensordot/MatMulMatMul5window_attention_1/dense_6/Tensordot/Reshape:output:0;window_attention_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
,window_attention_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2window_attention_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-window_attention_1/dense_6/Tensordot/concat_1ConcatV26window_attention_1/dense_6/Tensordot/GatherV2:output:05window_attention_1/dense_6/Tensordot/Const_2:output:0;window_attention_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:д
$window_attention_1/dense_6/TensordotReshape5window_attention_1/dense_6/Tensordot/MatMul:product:06window_attention_1/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ј
1window_attention_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Э
"window_attention_1/dense_6/BiasAddBiasAdd-window_attention_1/dense_6/Tensordot:output:09window_attention_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@q
,window_attention_1/dropout_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?Ы
*window_attention_1/dropout_3/dropout_1/MulMul+window_attention_1/dense_6/BiasAdd:output:05window_attention_1/dropout_3/dropout_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@
,window_attention_1/dropout_3/dropout_1/ShapeShape+window_attention_1/dense_6/BiasAdd:output:0*
T0*
_output_shapes
:Ю
Cwindow_attention_1/dropout_3/dropout_1/random_uniform/RandomUniformRandomUniform5window_attention_1/dropout_3/dropout_1/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@*
dtype0z
5window_attention_1/dropout_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<
3window_attention_1/dropout_3/dropout_1/GreaterEqualGreaterEqualLwindow_attention_1/dropout_3/dropout_1/random_uniform/RandomUniform:output:0>window_attention_1/dropout_3/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Б
+window_attention_1/dropout_3/dropout_1/CastCast7window_attention_1/dropout_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ@Ъ
,window_attention_1/dropout_3/dropout_1/Mul_1Mul.window_attention_1/dropout_3/dropout_1/Mul:z:0/window_attention_1/dropout_3/dropout_1/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   
	Reshape_4Reshape0window_attention_1/dropout_3/dropout_1/Mul_1:z:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @]
Roll_1/shiftConst*
_output_shapes
:*
dtype0*
valueB"      \
Roll_1/axisConst*
_output_shapes
:*
dtype0*
valueB"      Є
Roll_1RollReshape_6:output:0Roll_1/shift:output:0Roll_1/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:џџџџџџџџџ  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   v
	Reshape_7ReshapeRoll_1:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@S
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
valueB:
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
value	B :д
 drop_path_1/random_uniform/shapePack"drop_path_1/strided_slice:output:0+drop_path_1/random_uniform/shape/1:output:0+drop_path_1/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:Ї
(drop_path_1/random_uniform/RandomUniformRandomUniform)drop_path_1/random_uniform/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0V
drop_path_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path_1/addAddV2drop_path_1/add/x:output:01drop_path_1/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџe
drop_path_1/FloorFloordrop_path_1/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
drop_path_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path_1/truedivRealDivReshape_7:output:0drop_path_1/truediv/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@}
drop_path_1/mulMuldrop_path_1/truediv:z:0drop_path_1/Floor:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@[
addAddV2xdrop_path_1/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:К
"layer_normalization_3/moments/meanMeanadd:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЙ
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_3/batchnorm/mul_1Muladd:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ѕ
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: ­
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Г
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : №
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:И
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ч
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Щ
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЪ
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџq
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ћ
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ф
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Н
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџi
$sequential_1/activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?З
"sequential_1/activation_1/Gelu/mulMul-sequential_1/activation_1/Gelu/mul/x:output:0%sequential_1/dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџj
%sequential_1/activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?Р
&sequential_1/activation_1/Gelu/truedivRealDiv%sequential_1/dense_7/BiasAdd:output:0.sequential_1/activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
"sequential_1/activation_1/Gelu/ErfErf*sequential_1/activation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџi
$sequential_1/activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?К
"sequential_1/activation_1/Gelu/addAddV2-sequential_1/activation_1/Gelu/add/x:output:0&sequential_1/activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџГ
$sequential_1/activation_1/Gelu/mul_1Mul&sequential_1/activation_1/Gelu/mul:z:0&sequential_1/activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџi
$sequential_1/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?К
"sequential_1/dropout_4/dropout/MulMul(sequential_1/activation_1/Gelu/mul_1:z:0-sequential_1/dropout_4/dropout/Const:output:0*
T0*-
_output_shapes
:џџџџџџџџџ|
$sequential_1/dropout_4/dropout/ShapeShape(sequential_1/activation_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:Р
;sequential_1/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_4/dropout/Shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ*
dtype0r
-sequential_1/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<ё
+sequential_1/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_4/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:џџџџџџџџџЃ
#sequential_1/dropout_4/dropout/CastCast/sequential_1/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:џџџџџџџџџД
$sequential_1/dropout_4/dropout/Mul_1Mul&sequential_1/dropout_4/dropout/Mul:z:0'sequential_1/dropout_4/dropout/Cast:y:0*
T0*-
_output_shapes
:џџџџџџџџџЅ
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: ­
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Г
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : №
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:И
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ч
(sequential_1/dense_8/Tensordot/transpose	Transpose(sequential_1/dropout_4/dropout/Mul_1:z:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџЩ
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЩ
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ћ
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:У
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0М
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@i
$sequential_1/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?Ж
"sequential_1/dropout_5/dropout/MulMul%sequential_1/dense_8/BiasAdd:output:0-sequential_1/dropout_5/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@y
$sequential_1/dropout_5/dropout/ShapeShape%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:П
;sequential_1/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_5/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
dtype0r
-sequential_1/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<№
+sequential_1/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_5/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
#sequential_1/dropout_5/dropout/CastCast/sequential_1/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ@Г
$sequential_1/dropout_5/dropout/Mul_1Mul&sequential_1/dropout_5/dropout/Mul:z:0'sequential_1/dropout_5/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@k
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
valueB:
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
value	B :м
"drop_path_1/random_uniform_1/shapePack$drop_path_1/strided_slice_1:output:0-drop_path_1/random_uniform_1/shape/1:output:0-drop_path_1/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ћ
*drop_path_1/random_uniform_1/RandomUniformRandomUniform+drop_path_1/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0X
drop_path_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?Ѓ
drop_path_1/add_1AddV2drop_path_1/add_1/x:output:03drop_path_1/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџi
drop_path_1/Floor_1Floordrop_path_1/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ\
drop_path_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?Ѓ
drop_path_1/truediv_1RealDiv(sequential_1/dropout_5/dropout/Mul_1:z:0 drop_path_1/truediv_1/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
drop_path_1/mul_1Muldrop_path_1/truediv_1:z:0drop_path_1/Floor_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@e
add_1AddV2add:z:0drop_path_1/mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@
NoOpNoOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp/^window_attention_1/ExpandDims_1/ReadVariableOp^window_attention_1/Gather,^window_attention_1/Reshape_1/ReadVariableOp2^window_attention_1/dense_5/BiasAdd/ReadVariableOp4^window_attention_1/dense_5/Tensordot/ReadVariableOp2^window_attention_1/dense_6/BiasAdd/ReadVariableOp4^window_attention_1/dense_6/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ@: : : : : : : : : : : : : : : 2`
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
:џџџџџџџџџ@

_user_specified_namex
Д
ш
5map_while_stateless_random_flip_left_right_true_12091v
rmap_while_stateless_random_flip_left_right_reversev2_map_while_stateless_random_flip_left_right_control_dependency7
3map_while_stateless_random_flip_left_right_identity
9map/while/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:І
4map/while/stateless_random_flip_left_right/ReverseV2	ReverseV2rmap_while_stateless_random_flip_left_right_reversev2_map_while_stateless_random_flip_left_right_control_dependencyBmap/while/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*"
_output_shapes
:@@Ћ
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

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
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
є/

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
§џџџџџџџџy
&random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџp
&random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
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
ўџџџџџџџџ{
(random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџr
(random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
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
value	B :@
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
 *  B
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
value	B :@
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
 *  B
random_crop/cond/truediv_1RealDivrandom_crop/cond/Cast_2:y:0%random_crop/cond/truediv_1/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_3Castrandom_crop/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 
random_crop/cond/MinimumMinimum'random_crop/cond/strided_slice:output:0random_crop/cond/Cast_1:y:0*
T0*
_output_shapes
: 
random_crop/cond/Minimum_1Minimum)random_crop/cond/strided_slice_1:output:0random_crop/cond/Cast_3:y:0*
T0*
_output_shapes
: 
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
 *   @
random_crop/cond/truediv_2RealDivrandom_crop/cond/Cast_4:y:0%random_crop/cond/truediv_2/y:output:0*
T0*
_output_shapes
: o
random_crop/cond/Cast_5Castrandom_crop/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: 
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
 *   @
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
value	B : Ь
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
џџџџџџџџџe
random_crop/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџж
random_crop/cond/stack_1Pack#random_crop/cond/stack_1/0:output:0random_crop/cond/Minimum:z:0random_crop/cond/Minimum_1:z:0#random_crop/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:в
random_crop/cond/SliceSlicerandom_crop_cond_shape_inputsrandom_crop/cond/stack:output:0!random_crop/cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"џџџџџџџџџ@@џџџџџџџџџm
random_crop/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   д
&random_crop/cond/resize/ResizeBilinearResizeBilinearrandom_crop/cond/Slice:output:0%random_crop/cond/resize/size:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
half_pixel_centers(
random_crop/cond/IdentityIdentity7random_crop/cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:џџџџџџџџџ@@"?
random_crop_cond_identity"random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: :5 1
/
_output_shapes
:џџџџџџџџџ@@

b
)__inference_dropout_5_layer_call_fn_13353

inputs
identityЂStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12545t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
д
њ
B__inference_dense_8_layer_call_and_return_conditional_losses_13343

inputs4
!tensordot_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г
в
,__inference_sequential_1_layer_call_fn_12926

inputs
unknown:	@
	unknown_0:	
	unknown_1:	@
	unknown_2:@
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_12629t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
и
ћ
B__inference_dense_3_layer_call_and_return_conditional_losses_12162

inputs4
!tensordot_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџe
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ЮG
й
G__inference_sequential_1_layer_call_and_return_conditional_losses_12992

inputs<
)dense_7_tensordot_readvariableop_resource:	@6
'dense_7_biasadd_readvariableop_resource:	<
)dense_8_tensordot_readvariableop_resource:	@5
'dense_8_biasadd_readvariableop_resource:@
identityЂdense_7/BiasAdd/ReadVariableOpЂ dense_7/Tensordot/ReadVariableOpЂdense_8/BiasAdd/ReadVariableOpЂ dense_8/Tensordot/ReadVariableOp
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : л
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
value	B : п
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
valueB: 
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_7/Tensordot/transpose	Transposeinputs!dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЃ
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџ\
activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
activation_1/Gelu/mulMul activation_1/Gelu/mul/x:output:0dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџ]
activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?
activation_1/Gelu/truedivRealDivdense_7/BiasAdd:output:0!activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџs
activation_1/Gelu/ErfErfactivation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџ\
activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
activation_1/Gelu/addAddV2 activation_1/Gelu/add/x:output:0activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџ
activation_1/Gelu/mul_1Mulactivation_1/Gelu/mul:z:0activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџs
dropout_4/IdentityIdentityactivation_1/Gelu/mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџ
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : л
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
value	B : п
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
valueB: 
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
dense_8/Tensordot/transpose	Transposedropout_4/Identity:output:0!dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџЂ
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЂ
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@o
dropout_5/IdentityIdentitydense_8/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@o
IdentityIdentitydropout_5/Identity:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@Ю
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ђ

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_12297

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?j
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Ќ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:џџџџџџџџџu
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:џџџџџџџџџo
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:џџџџџџџџџ_
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


c
D__inference_dropout_5_layer_call_and_return_conditional_losses_13370

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Ћ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Џ
а
*__inference_sequential_layer_call_fn_12741

inputs
unknown:	@
	unknown_0:	
	unknown_1:	@
	unknown_2:@
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12233t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
њ(
ъ
I__inference_patch_embedding_layer_call_and_return_conditional_losses_2689	
patch9
'dense_tensordot_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@4
!embedding_embedding_lookup_372620:	@
identityЂdense/BiasAdd/ReadVariableOpЂdense/Tensordot/ReadVariableOpЂembedding/embedding_lookupM
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
B :M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :m
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:
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
value	B : г
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
value	B : з
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
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposepatchdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : П
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@г
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_372620range:output:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/372620*
_output_shapes
:	@*
dtype0Д
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/372620*
_output_shapes
:	@
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	@
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ѓ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 [
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:S O
,
_output_shapes
:џџџџџџџџџ

_user_specified_namepatch
Ж
{
+__inference_random_flip_layer_call_fn_12012

inputs
unknown:	
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10305w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
	
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
:џџџџџџџџџP
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?m
Gelu/truedivRealDivinputsGelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџY
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџO

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџe

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџ\
IdentityIdentityGelu/mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ§

L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_6078
xI
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@O
<window_attention_1_dense_5_tensordot_readvariableop_resource:	@РI
:window_attention_1_dense_5_biasadd_readvariableop_resource:	РF
4window_attention_1_reshape_1_readvariableop_resource:	4
"window_attention_1_gather_resource:	N
7window_attention_1_expanddims_1_readvariableop_resource:N
<window_attention_1_dense_6_tensordot_readvariableop_resource:@@H
:window_attention_1_dense_6_biasadd_readvariableop_resource:@I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_3_batchnorm_readvariableop_resource:@I
6sequential_1_dense_7_tensordot_readvariableop_resource:	@C
4sequential_1_dense_7_biasadd_readvariableop_resource:	I
6sequential_1_dense_8_tensordot_readvariableop_resource:	@B
4sequential_1_dense_8_biasadd_readvariableop_resource:@
identityЂ.layer_normalization_2/batchnorm/ReadVariableOpЂ2layer_normalization_2/batchnorm/mul/ReadVariableOpЂ.layer_normalization_3/batchnorm/ReadVariableOpЂ2layer_normalization_3/batchnorm/mul/ReadVariableOpЂ+sequential_1/dense_7/BiasAdd/ReadVariableOpЂ-sequential_1/dense_7/Tensordot/ReadVariableOpЂ+sequential_1/dense_8/BiasAdd/ReadVariableOpЂ-sequential_1/dense_8/Tensordot/ReadVariableOpЂ.window_attention_1/ExpandDims_1/ReadVariableOpЂwindow_attention_1/GatherЂ+window_attention_1/Reshape_1/ReadVariableOpЂ1window_attention_1/dense_5/BiasAdd/ReadVariableOpЂ3window_attention_1/dense_5/Tensordot/ReadVariableOpЂ1window_attention_1/dense_6/BiasAdd/ReadVariableOpЂ3window_attention_1/dense_6/Tensordot/ReadVariableOp~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Д
"layer_normalization_2/moments/meanMeanx=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџГ
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencex3layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_2/batchnorm/mul_1Mulx'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   
ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @[

Roll/shiftConst*
_output_shapes
:*
dtype0*
valueB"џџџџџџџџZ
	Roll/axisConst*
_output_shapes
:*
dtype0*
valueB"      
RollRollReshape:output:0Roll/shift:output:0Roll/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:џџџџџџџџџ  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_1ReshapeRoll:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Б
3window_attention_1/dense_5/Tensordot/ReadVariableOpReadVariableOp<window_attention_1_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	@Р*
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
value	B : Ї
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
value	B : Ћ
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
valueB: П
)window_attention_1/dense_5/Tensordot/ProdProd6window_attention_1/dense_5/Tensordot/GatherV2:output:03window_attention_1/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Х
+window_attention_1/dense_5/Tensordot/Prod_1Prod8window_attention_1/dense_5/Tensordot/GatherV2_1:output:05window_attention_1/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention_1/dense_5/Tensordot/concatConcatV22window_attention_1/dense_5/Tensordot/free:output:02window_attention_1/dense_5/Tensordot/axes:output:09window_attention_1/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ъ
*window_attention_1/dense_5/Tensordot/stackPack2window_attention_1/dense_5/Tensordot/Prod:output:04window_attention_1/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Л
.window_attention_1/dense_5/Tensordot/transpose	TransposeReshape_3:output:04window_attention_1/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@л
,window_attention_1/dense_5/Tensordot/ReshapeReshape2window_attention_1/dense_5/Tensordot/transpose:y:03window_attention_1/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџм
+window_attention_1/dense_5/Tensordot/MatMulMatMul5window_attention_1/dense_5/Tensordot/Reshape:output:0;window_attention_1/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџРw
,window_attention_1/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Рt
2window_attention_1/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-window_attention_1/dense_5/Tensordot/concat_1ConcatV26window_attention_1/dense_5/Tensordot/GatherV2:output:05window_attention_1/dense_5/Tensordot/Const_2:output:0;window_attention_1/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:е
$window_attention_1/dense_5/TensordotReshape5window_attention_1/dense_5/Tensordot/MatMul:product:06window_attention_1/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџРЉ
1window_attention_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype0Ю
"window_attention_1/dense_5/BiasAddBiasAdd-window_attention_1/dense_5/Tensordot:output:09window_attention_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџР}
 window_attention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Л
window_attention_1/ReshapeReshape+window_attention_1/dense_5/BiasAdd:output:0)window_attention_1/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ~
!window_attention_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                И
window_attention_1/transpose	Transpose#window_attention_1/Reshape:output:0*window_attention_1/transpose/perm:output:0*
T0*3
_output_shapes!
:џџџџџџџџџp
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
valueB:Ш
 window_attention_1/strided_sliceStridedSlice window_attention_1/transpose:y:0/window_attention_1/strided_slice/stack:output:01window_attention_1/strided_slice/stack_1:output:01window_attention_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:а
"window_attention_1/strided_slice_1StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_1/stack:output:03window_attention_1/strided_slice_1/stack_1:output:03window_attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:а
"window_attention_1/strided_slice_2StridedSlice window_attention_1/transpose:y:01window_attention_1/strided_slice_2/stack:output:03window_attention_1/strided_slice_2/stack_1:output:03window_attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask]
window_attention_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ>Ѕ
window_attention_1/mulMul)window_attention_1/strided_slice:output:0!window_attention_1/mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|
#window_attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             Р
window_attention_1/transpose_1	Transpose+window_attention_1/strided_slice_1:output:0,window_attention_1/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџЄ
window_attention_1/matmulBatchMatMulV2window_attention_1/mul:z:0"window_attention_1/transpose_1:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 
+window_attention_1/Reshape_1/ReadVariableOpReadVariableOp4window_attention_1_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	u
"window_attention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЎ
window_attention_1/Reshape_1Reshape3window_attention_1/Reshape_1/ReadVariableOp:value:0+window_attention_1/Reshape_1/shape:output:0*
T0	*
_output_shapes
:Г
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
valueB"      џџџџЇ
window_attention_1/Reshape_2Reshape$window_attention_1/Identity:output:0+window_attention_1/Reshape_2/shape:output:0*
T0*"
_output_shapes
:x
#window_attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
window_attention_1/transpose_2	Transpose%window_attention_1/Reshape_2:output:0,window_attention_1/transpose_2/perm:output:0*
T0*"
_output_shapes
:c
!window_attention_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ќ
window_attention_1/ExpandDims
ExpandDims"window_attention_1/transpose_2:y:0*window_attention_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:Ѕ
window_attention_1/addAddV2"window_attention_1/matmul:output:0&window_attention_1/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџЋ
.window_attention_1/ExpandDims_1/ReadVariableOpReadVariableOp7window_attention_1_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype0e
#window_attention_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :Х
window_attention_1/ExpandDims_1
ExpandDims6window_attention_1/ExpandDims_1/ReadVariableOp:value:0,window_attention_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:e
#window_attention_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : Л
window_attention_1/ExpandDims_2
ExpandDims(window_attention_1/ExpandDims_1:output:0,window_attention_1/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:
window_attention_1/CastCast(window_attention_1/ExpandDims_2:output:0*

DstT0*

SrcT0*+
_output_shapes
:
"window_attention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Џ
window_attention_1/Reshape_3Reshapewindow_attention_1/add:z:0+window_attention_1/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџЄ
window_attention_1/add_1AddV2%window_attention_1/Reshape_3:output:0window_attention_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџ{
"window_attention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         Ќ
window_attention_1/Reshape_4Reshapewindow_attention_1/add_1:z:0+window_attention_1/Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
window_attention_1/SoftmaxSoftmax%window_attention_1/Reshape_4:output:0*
T0*/
_output_shapes
:џџџџџџџџџo
*window_attention_1/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?Ф
(window_attention_1/dropout_3/dropout/MulMul$window_attention_1/Softmax:softmax:03window_attention_1/dropout_3/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ~
*window_attention_1/dropout_3/dropout/ShapeShape$window_attention_1/Softmax:softmax:0*
T0*
_output_shapes
:Ю
Awindow_attention_1/dropout_3/dropout/random_uniform/RandomUniformRandomUniform3window_attention_1/dropout_3/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0x
3window_attention_1/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<
1window_attention_1/dropout_3/dropout/GreaterEqualGreaterEqualJwindow_attention_1/dropout_3/dropout/random_uniform/RandomUniform:output:0<window_attention_1/dropout_3/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџБ
)window_attention_1/dropout_3/dropout/CastCast5window_attention_1/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџШ
*window_attention_1/dropout_3/dropout/Mul_1Mul,window_attention_1/dropout_3/dropout/Mul:z:0-window_attention_1/dropout_3/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџУ
window_attention_1/matmul_1BatchMatMulV2.window_attention_1/dropout_3/dropout/Mul_1:z:0+window_attention_1/strided_slice_2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|
#window_attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Й
window_attention_1/transpose_3	Transpose$window_attention_1/matmul_1:output:0,window_attention_1/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
"window_attention_1/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   Ў
window_attention_1/Reshape_5Reshape"window_attention_1/transpose_3:y:0+window_attention_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@А
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
value	B : Ї
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
value	B : Ћ
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
valueB: П
)window_attention_1/dense_6/Tensordot/ProdProd6window_attention_1/dense_6/Tensordot/GatherV2:output:03window_attention_1/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,window_attention_1/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Х
+window_attention_1/dense_6/Tensordot/Prod_1Prod8window_attention_1/dense_6/Tensordot/GatherV2_1:output:05window_attention_1/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0window_attention_1/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention_1/dense_6/Tensordot/concatConcatV22window_attention_1/dense_6/Tensordot/free:output:02window_attention_1/dense_6/Tensordot/axes:output:09window_attention_1/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ъ
*window_attention_1/dense_6/Tensordot/stackPack2window_attention_1/dense_6/Tensordot/Prod:output:04window_attention_1/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ю
.window_attention_1/dense_6/Tensordot/transpose	Transpose%window_attention_1/Reshape_5:output:04window_attention_1/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@л
,window_attention_1/dense_6/Tensordot/ReshapeReshape2window_attention_1/dense_6/Tensordot/transpose:y:03window_attention_1/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџл
+window_attention_1/dense_6/Tensordot/MatMulMatMul5window_attention_1/dense_6/Tensordot/Reshape:output:0;window_attention_1/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
,window_attention_1/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2window_attention_1/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-window_attention_1/dense_6/Tensordot/concat_1ConcatV26window_attention_1/dense_6/Tensordot/GatherV2:output:05window_attention_1/dense_6/Tensordot/Const_2:output:0;window_attention_1/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:д
$window_attention_1/dense_6/TensordotReshape5window_attention_1/dense_6/Tensordot/MatMul:product:06window_attention_1/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ј
1window_attention_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp:window_attention_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Э
"window_attention_1/dense_6/BiasAddBiasAdd-window_attention_1/dense_6/Tensordot:output:09window_attention_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@q
,window_attention_1/dropout_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?Ы
*window_attention_1/dropout_3/dropout_1/MulMul+window_attention_1/dense_6/BiasAdd:output:05window_attention_1/dropout_3/dropout_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@
,window_attention_1/dropout_3/dropout_1/ShapeShape+window_attention_1/dense_6/BiasAdd:output:0*
T0*
_output_shapes
:Ю
Cwindow_attention_1/dropout_3/dropout_1/random_uniform/RandomUniformRandomUniform5window_attention_1/dropout_3/dropout_1/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@*
dtype0z
5window_attention_1/dropout_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<
3window_attention_1/dropout_3/dropout_1/GreaterEqualGreaterEqualLwindow_attention_1/dropout_3/dropout_1/random_uniform/RandomUniform:output:0>window_attention_1/dropout_3/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Б
+window_attention_1/dropout_3/dropout_1/CastCast7window_attention_1/dropout_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ@Ъ
,window_attention_1/dropout_3/dropout_1/Mul_1Mul.window_attention_1/dropout_3/dropout_1/Mul:z:0/window_attention_1/dropout_3/dropout_1/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   
	Reshape_4Reshape0window_attention_1/dropout_3/dropout_1/Mul_1:z:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @]
Roll_1/shiftConst*
_output_shapes
:*
dtype0*
valueB"      \
Roll_1/axisConst*
_output_shapes
:*
dtype0*
valueB"      Є
Roll_1RollReshape_6:output:0Roll_1/shift:output:0Roll_1/axis:output:0*
T0*
Taxis0*
Tshift0*/
_output_shapes
:џџџџџџџџџ  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   v
	Reshape_7ReshapeRoll_1:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@S
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
valueB:
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
value	B :д
 drop_path_1/random_uniform/shapePack"drop_path_1/strided_slice:output:0+drop_path_1/random_uniform/shape/1:output:0+drop_path_1/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:Ї
(drop_path_1/random_uniform/RandomUniformRandomUniform)drop_path_1/random_uniform/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0V
drop_path_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path_1/addAddV2drop_path_1/add/x:output:01drop_path_1/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџe
drop_path_1/FloorFloordrop_path_1/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
drop_path_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path_1/truedivRealDivReshape_7:output:0drop_path_1/truediv/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@}
drop_path_1/mulMuldrop_path_1/truediv:z:0drop_path_1/Floor:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@[
addAddV2xdrop_path_1/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:К
"layer_normalization_3/moments/meanMeanadd:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЙ
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_3/batchnorm/mul_1Muladd:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ѕ
-sequential_1/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: ­
#sequential_1/dense_7/Tensordot/ProdProd0sequential_1/dense_7/Tensordot/GatherV2:output:0-sequential_1/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Г
%sequential_1/dense_7/Tensordot/Prod_1Prod2sequential_1/dense_7/Tensordot/GatherV2_1:output:0/sequential_1/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : №
%sequential_1/dense_7/Tensordot/concatConcatV2,sequential_1/dense_7/Tensordot/free:output:0,sequential_1/dense_7/Tensordot/axes:output:03sequential_1/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:И
$sequential_1/dense_7/Tensordot/stackPack,sequential_1/dense_7/Tensordot/Prod:output:0.sequential_1/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ч
(sequential_1/dense_7/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0.sequential_1/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Щ
&sequential_1/dense_7/Tensordot/ReshapeReshape,sequential_1/dense_7/Tensordot/transpose:y:0-sequential_1/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЪ
%sequential_1/dense_7/Tensordot/MatMulMatMul/sequential_1/dense_7/Tensordot/Reshape:output:05sequential_1/dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџq
&sequential_1/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ћ
'sequential_1/dense_7/Tensordot/concat_1ConcatV20sequential_1/dense_7/Tensordot/GatherV2:output:0/sequential_1/dense_7/Tensordot/Const_2:output:05sequential_1/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ф
sequential_1/dense_7/TensordotReshape/sequential_1/dense_7/Tensordot/MatMul:product:00sequential_1/dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Н
sequential_1/dense_7/BiasAddBiasAdd'sequential_1/dense_7/Tensordot:output:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџi
$sequential_1/activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?З
"sequential_1/activation_1/Gelu/mulMul-sequential_1/activation_1/Gelu/mul/x:output:0%sequential_1/dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџj
%sequential_1/activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?Р
&sequential_1/activation_1/Gelu/truedivRealDiv%sequential_1/dense_7/BiasAdd:output:0.sequential_1/activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
"sequential_1/activation_1/Gelu/ErfErf*sequential_1/activation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџi
$sequential_1/activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?К
"sequential_1/activation_1/Gelu/addAddV2-sequential_1/activation_1/Gelu/add/x:output:0&sequential_1/activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџГ
$sequential_1/activation_1/Gelu/mul_1Mul&sequential_1/activation_1/Gelu/mul:z:0&sequential_1/activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџi
$sequential_1/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?К
"sequential_1/dropout_4/dropout/MulMul(sequential_1/activation_1/Gelu/mul_1:z:0-sequential_1/dropout_4/dropout/Const:output:0*
T0*-
_output_shapes
:џџџџџџџџџ|
$sequential_1/dropout_4/dropout/ShapeShape(sequential_1/activation_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:Р
;sequential_1/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_4/dropout/Shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ*
dtype0r
-sequential_1/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<ё
+sequential_1/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_4/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:џџџџџџџџџЃ
#sequential_1/dropout_4/dropout/CastCast/sequential_1/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:џџџџџџџџџД
$sequential_1/dropout_4/dropout/Mul_1Mul&sequential_1/dropout_4/dropout/Mul:z:0'sequential_1/dropout_4/dropout/Cast:y:0*
T0*-
_output_shapes
:џџџџџџџџџЅ
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: ­
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Г
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : №
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:И
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ч
(sequential_1/dense_8/Tensordot/transpose	Transpose(sequential_1/dropout_4/dropout/Mul_1:z:0.sequential_1/dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџЩ
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЩ
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ћ
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:У
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0М
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@i
$sequential_1/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?Ж
"sequential_1/dropout_5/dropout/MulMul%sequential_1/dense_8/BiasAdd:output:0-sequential_1/dropout_5/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@y
$sequential_1/dropout_5/dropout/ShapeShape%sequential_1/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:П
;sequential_1/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_5/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
dtype0r
-sequential_1/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<№
+sequential_1/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_5/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
#sequential_1/dropout_5/dropout/CastCast/sequential_1/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ@Г
$sequential_1/dropout_5/dropout/Mul_1Mul&sequential_1/dropout_5/dropout/Mul:z:0'sequential_1/dropout_5/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@k
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
valueB:
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
value	B :м
"drop_path_1/random_uniform_1/shapePack$drop_path_1/strided_slice_1:output:0-drop_path_1/random_uniform_1/shape/1:output:0-drop_path_1/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ћ
*drop_path_1/random_uniform_1/RandomUniformRandomUniform+drop_path_1/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0X
drop_path_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?Ѓ
drop_path_1/add_1AddV2drop_path_1/add_1/x:output:03drop_path_1/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџi
drop_path_1/Floor_1Floordrop_path_1/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ\
drop_path_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?Ѓ
drop_path_1/truediv_1RealDiv(sequential_1/dropout_5/dropout/Mul_1:z:0 drop_path_1/truediv_1/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
drop_path_1/mul_1Muldrop_path_1/truediv_1:z:0drop_path_1/Floor_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@e
add_1AddV2add:z:0drop_path_1/mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@
NoOpNoOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp.^sequential_1/dense_7/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp/^window_attention_1/ExpandDims_1/ReadVariableOp^window_attention_1/Gather,^window_attention_1/Reshape_1/ReadVariableOp2^window_attention_1/dense_5/BiasAdd/ReadVariableOp4^window_attention_1/dense_5/Tensordot/ReadVariableOp2^window_attention_1/dense_6/BiasAdd/ReadVariableOp4^window_attention_1/dense_6/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ@: : : : : : : : : : : : : : : 2`
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
:џџџџџџџџџ@

_user_specified_namex
ЌG
з
E__inference_sequential_layer_call_and_return_conditional_losses_12820

inputs<
)dense_3_tensordot_readvariableop_resource:	@6
'dense_3_biasadd_readvariableop_resource:	<
)dense_4_tensordot_readvariableop_resource:	@5
'dense_4_biasadd_readvariableop_resource:@
identityЂdense_3/BiasAdd/ReadVariableOpЂ dense_3/Tensordot/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂ dense_4/Tensordot/ReadVariableOp
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : л
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
value	B : п
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
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/transpose	Transposeinputs!dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЃ
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџZ
activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
activation/Gelu/mulMulactivation/Gelu/mul/x:output:0dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџ[
activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?
activation/Gelu/truedivRealDivdense_3/BiasAdd:output:0activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџo
activation/Gelu/ErfErfactivation/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџZ
activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
activation/Gelu/addAddV2activation/Gelu/add/x:output:0activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџ
activation/Gelu/mul_1Mulactivation/Gelu/mul:z:0activation/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџq
dropout_1/IdentityIdentityactivation/Gelu/mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџ
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : л
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
value	B : п
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
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
dense_4/Tensordot/transpose	Transposedropout_1/Identity:output:0!dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџЂ
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЂ
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@o
dropout_2/IdentityIdentitydense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@o
IdentityIdentitydropout_2/Identity:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@Ю
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Л
і
E__inference_sequential_layer_call_and_return_conditional_losses_12406
dense_3_input 
dense_3_12392:	@
dense_3_12394:	 
dense_4_12399:	@
dense_4_12401:@
identityЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallі
dense_3/StatefulPartitionedCallStatefulPartitionedCalldense_3_inputdense_3_12392dense_3_12394*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_12162у
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12180ь
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12297
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_4_12399dense_4_12401*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_12219
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12264~
IdentityIdentity*dropout_2/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@в
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ@
'
_user_specified_namedense_3_input


1__inference_swin_transformer_1_layer_call_fn_1623
x
unknown:@
	unknown_0:@
	unknown_1:	@Р
	unknown_2:	Р
	unknown_3:	
	unknown_4:	 
	unknown_5:
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:	@

unknown_11:	

unknown_12:	@

unknown_13:@
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_1603`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ@: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex

И
.__inference_patch_embedding_layer_call_fn_1272	
patch
unknown:@
	unknown_0:@
	unknown_1:	@
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallpatchunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_patch_embedding_layer_call_and_return_conditional_losses_1264`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:џџџџџџџџџ

_user_specified_namepatch
Т
H
,__inference_patch_extract_layer_call_fn_3752

images
identityМ
PartitionedCallPartitionedCallimages*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_patch_extract_layer_call_and_return_conditional_losses_3697e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameimages
Ч
Ў
E__inference_sequential_layer_call_and_return_conditional_losses_12389
dense_3_input 
dense_3_12375:	@
dense_3_12377:	 
dense_4_12382:	@
dense_4_12384:@
identityЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallі
dense_3/StatefulPartitionedCallStatefulPartitionedCalldense_3_inputdense_3_12375dense_3_12377*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_12162у
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12180м
dropout_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12187
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_4_12382dense_4_12384*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_12219р
dropout_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12230v
IdentityIdentity"dropout_2/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ@
'
_user_specified_namedense_3_input

b
)__inference_dropout_2_layer_call_fn_13204

inputs
identityЂStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12264t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
с
ћ
(__inference_restored_function_body_10686
x
unknown:@
	unknown_0:@
	unknown_1:	@Р
	unknown_2:	Р
	unknown_3:	
	unknown_4:	 
	unknown_5:
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:	@

unknown_11:	

unknown_12:	@

unknown_13:@
identityЂStatefulPartitionedCallя
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
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_6078t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ@: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
ы
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_13209

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџ@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

x
'__inference_restored_function_body_9927
x
unknown:

identityЂStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallxunknown*
Tin
2*
Tout
2*-
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_patch_merging_layer_call_and_return_conditional_losses_2938u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:џџџџџџџџџ@: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
ј
З
%__inference_model_layer_call_fn_10178
input_1
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:@
	unknown_3:@
	unknown_4:	@Р
	unknown_5:	Р
	unknown_6:	
	unknown_7:	
	unknown_8:@@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:	@

unknown_13:	

unknown_14:	@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@Р

unknown_19:	Р

unknown_20:	

unknown_21:	!

unknown_22:

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:	@

unknown_28:	

unknown_29:	@

unknown_30:@

unknown_31:


unknown_32:	W

unknown_33:W
identityЂStatefulPartitionedCall
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
:џџџџџџџџџW*E
_read_only_resource_inputs'
%#	
 !"#*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџW`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@@
!
_user_specified_name	input_1
Г
в
,__inference_sequential_1_layer_call_fn_12913

inputs
unknown:	@
	unknown_0:	
	unknown_1:	@
	unknown_2:@
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_12514t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ќ
и
(__inference_restored_function_body_10622
x
unknown:@
	unknown_0:@
	unknown_1:	@Р
	unknown_2:	Р
	unknown_3:	
	unknown_4:	
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:	@

unknown_10:	

unknown_11:	@

unknown_12:@
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*,
_output_shapes
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2087t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
Я
А
G__inference_sequential_1_layer_call_and_return_conditional_losses_12670
dense_7_input 
dense_7_12656:	@
dense_7_12658:	 
dense_8_12663:	@
dense_8_12665:@
identityЂdense_7/StatefulPartitionedCallЂdense_8/StatefulPartitionedCallі
dense_7/StatefulPartitionedCallStatefulPartitionedCalldense_7_inputdense_7_12656dense_7_12658*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_12443ч
activation_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_12461о
dropout_4/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12468
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_8_12663dense_8_12665*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_12500р
dropout_5/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12511v
IdentityIdentity"dropout_5/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ@
'
_user_specified_namedense_7_input
Ч
ъ
%__inference_model_layer_call_fn_11285

inputs
unknown:	
	unknown_0:	
	unknown_1:@
	unknown_2:@
	unknown_3:	@
	unknown_4:@
	unknown_5:@
	unknown_6:	@Р
	unknown_7:	Р
	unknown_8:	
	unknown_9:	

unknown_10:@@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:	@

unknown_15:	

unknown_16:	@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:	@Р

unknown_21:	Р

unknown_22:	

unknown_23:	!

unknown_24:

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:	@

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:


unknown_34:	W

unknown_35:W
identityЂStatefulPartitionedCallД
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
:џџџџџџџџџW*E
_read_only_resource_inputs'
%#	
 !"#$%*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10728o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџW`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
К
б
J__inference_swin_transformer_layer_call_and_return_conditional_losses_6423
xG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@M
:window_attention_dense_1_tensordot_readvariableop_resource:	@РG
8window_attention_dense_1_biasadd_readvariableop_resource:	РD
2window_attention_reshape_1_readvariableop_resource:	2
 window_attention_gather_resource:	L
:window_attention_dense_2_tensordot_readvariableop_resource:@@F
8window_attention_dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@G
4sequential_dense_3_tensordot_readvariableop_resource:	@A
2sequential_dense_3_biasadd_readvariableop_resource:	G
4sequential_dense_4_tensordot_readvariableop_resource:	@@
2sequential_dense_4_biasadd_readvariableop_resource:@
identityЂ,layer_normalization/batchnorm/ReadVariableOpЂ0layer_normalization/batchnorm/mul/ReadVariableOpЂ.layer_normalization_1/batchnorm/ReadVariableOpЂ2layer_normalization_1/batchnorm/mul/ReadVariableOpЂ)sequential/dense_3/BiasAdd/ReadVariableOpЂ+sequential/dense_3/Tensordot/ReadVariableOpЂ)sequential/dense_4/BiasAdd/ReadVariableOpЂ+sequential/dense_4/Tensordot/ReadVariableOpЂwindow_attention/GatherЂ)window_attention/Reshape_1/ReadVariableOpЂ/window_attention/dense_1/BiasAdd/ReadVariableOpЂ1window_attention/dense_1/Tensordot/ReadVariableOpЂ/window_attention/dense_2/BiasAdd/ReadVariableOpЂ1window_attention/dense_2/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:А
 layer_normalization/moments/meanMeanx;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЏ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencex1layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ш
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7О
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџІ
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Т
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
#layer_normalization/batchnorm/mul_1Mulx%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Г
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0О
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Г
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   
ReshapeReshape'layer_normalization/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_1ReshapeReshape:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@­
1window_attention/dense_1/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@Р*
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
value	B : 
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
value	B : Ѓ
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
valueB: Й
'window_attention/dense_1/Tensordot/ProdProd4window_attention/dense_1/Tensordot/GatherV2:output:01window_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
)window_attention/dense_1/Tensordot/Prod_1Prod6window_attention/dense_1/Tensordot/GatherV2_1:output:03window_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)window_attention/dense_1/Tensordot/concatConcatV20window_attention/dense_1/Tensordot/free:output:00window_attention/dense_1/Tensordot/axes:output:07window_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
(window_attention/dense_1/Tensordot/stackPack0window_attention/dense_1/Tensordot/Prod:output:02window_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:З
,window_attention/dense_1/Tensordot/transpose	TransposeReshape_3:output:02window_attention/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@е
*window_attention/dense_1/Tensordot/ReshapeReshape0window_attention/dense_1/Tensordot/transpose:y:01window_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџж
)window_attention/dense_1/Tensordot/MatMulMatMul3window_attention/dense_1/Tensordot/Reshape:output:09window_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџРu
*window_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Рr
0window_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention/dense_1/Tensordot/concat_1ConcatV24window_attention/dense_1/Tensordot/GatherV2:output:03window_attention/dense_1/Tensordot/Const_2:output:09window_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Я
"window_attention/dense_1/TensordotReshape3window_attention/dense_1/Tensordot/MatMul:product:04window_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџРЅ
/window_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype0Ш
 window_attention/dense_1/BiasAddBiasAdd+window_attention/dense_1/Tensordot:output:07window_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџР{
window_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Е
window_attention/ReshapeReshape)window_attention/dense_1/BiasAdd:output:0'window_attention/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
window_attention/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                В
window_attention/transpose	Transpose!window_attention/Reshape:output:0(window_attention/transpose/perm:output:0*
T0*3
_output_shapes!
:џџџџџџџџџn
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
valueB:О
window_attention/strided_sliceStridedSlicewindow_attention/transpose:y:0-window_attention/strided_slice/stack:output:0/window_attention/strided_slice/stack_1:output:0/window_attention/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:Ц
 window_attention/strided_slice_1StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_1/stack:output:01window_attention/strided_slice_1/stack_1:output:01window_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:Ц
 window_attention/strided_slice_2StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_2/stack:output:01window_attention/strided_slice_2/stack_1:output:01window_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask[
window_attention/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ>
window_attention/mulMul'window_attention/strided_slice:output:0window_attention/mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџz
!window_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             К
window_attention/transpose_1	Transpose)window_attention/strided_slice_1:output:0*window_attention/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
window_attention/matmulBatchMatMulV2window_attention/mul:z:0 window_attention/transpose_1:y:0*
T0*/
_output_shapes
:џџџџџџџџџ
)window_attention/Reshape_1/ReadVariableOpReadVariableOp2window_attention_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	s
 window_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЈ
window_attention/Reshape_1Reshape1window_attention/Reshape_1/ReadVariableOp:value:0)window_attention/Reshape_1/shape:output:0*
T0	*
_output_shapes
:­
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
valueB"      џџџџЁ
window_attention/Reshape_2Reshape"window_attention/Identity:output:0)window_attention/Reshape_2/shape:output:0*
T0*"
_output_shapes
:v
!window_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ї
window_attention/transpose_2	Transpose#window_attention/Reshape_2:output:0*window_attention/transpose_2/perm:output:0*
T0*"
_output_shapes
:a
window_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : І
window_attention/ExpandDims
ExpandDims window_attention/transpose_2:y:0(window_attention/ExpandDims/dim:output:0*
T0*&
_output_shapes
:
window_attention/addAddV2 window_attention/matmul:output:0$window_attention/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
window_attention/SoftmaxSoftmaxwindow_attention/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ
!window_attention/dropout/IdentityIdentity"window_attention/Softmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџЛ
window_attention/matmul_1BatchMatMulV2*window_attention/dropout/Identity:output:0)window_attention/strided_slice_2:output:0*
T0*/
_output_shapes
:џџџџџџџџџz
!window_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Г
window_attention/transpose_3	Transpose"window_attention/matmul_1:output:0*window_attention/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџu
 window_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   Ј
window_attention/Reshape_3Reshape window_attention/transpose_3:y:0)window_attention/Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ќ
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
value	B : 
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
value	B : Ѓ
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
valueB: Й
'window_attention/dense_2/Tensordot/ProdProd4window_attention/dense_2/Tensordot/GatherV2:output:01window_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
)window_attention/dense_2/Tensordot/Prod_1Prod6window_attention/dense_2/Tensordot/GatherV2_1:output:03window_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)window_attention/dense_2/Tensordot/concatConcatV20window_attention/dense_2/Tensordot/free:output:00window_attention/dense_2/Tensordot/axes:output:07window_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
(window_attention/dense_2/Tensordot/stackPack0window_attention/dense_2/Tensordot/Prod:output:02window_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ш
,window_attention/dense_2/Tensordot/transpose	Transpose#window_attention/Reshape_3:output:02window_attention/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@е
*window_attention/dense_2/Tensordot/ReshapeReshape0window_attention/dense_2/Tensordot/transpose:y:01window_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџе
)window_attention/dense_2/Tensordot/MatMulMatMul3window_attention/dense_2/Tensordot/Reshape:output:09window_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@t
*window_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@r
0window_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention/dense_2/Tensordot/concat_1ConcatV24window_attention/dense_2/Tensordot/GatherV2:output:03window_attention/dense_2/Tensordot/Const_2:output:09window_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ю
"window_attention/dense_2/TensordotReshape3window_attention/dense_2/Tensordot/MatMul:product:04window_attention/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Є
/window_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ч
 window_attention/dense_2/BiasAddBiasAdd+window_attention/dense_2/Tensordot:output:07window_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@
#window_attention/dropout/Identity_1Identity)window_attention/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   
	Reshape_4Reshape,window_attention/dropout/Identity_1:output:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   y
	Reshape_7ReshapeReshape_6:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Q
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
valueB:
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
value	B :Ь
drop_path/random_uniform/shapePack drop_path/strided_slice:output:0)drop_path/random_uniform/shape/1:output:0)drop_path/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:Ѓ
&drop_path/random_uniform/RandomUniformRandomUniform'drop_path/random_uniform/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0T
drop_path/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/addAddV2drop_path/add/x:output:0/drop_path/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџa
drop_path/FloorFloordrop_path/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
drop_path/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/truedivRealDivReshape_7:output:0drop_path/truediv/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@w
drop_path/mulMuldrop_path/truediv:z:0drop_path/Floor:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@Y
addAddV2xdrop_path/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:К
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЙ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ё
+sequential/dense_3/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: Ї
!sequential/dense_3/Tensordot/ProdProd.sequential/dense_3/Tensordot/GatherV2:output:0+sequential/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_3/Tensordot/Prod_1Prod0sequential/dense_3/Tensordot/GatherV2_1:output:0-sequential/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
#sequential/dense_3/Tensordot/concatConcatV2*sequential/dense_3/Tensordot/free:output:0*sequential/dense_3/Tensordot/axes:output:01sequential/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:В
"sequential/dense_3/Tensordot/stackPack*sequential/dense_3/Tensordot/Prod:output:0,sequential/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:У
&sequential/dense_3/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0,sequential/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@У
$sequential/dense_3/Tensordot/ReshapeReshape*sequential/dense_3/Tensordot/transpose:y:0+sequential/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџФ
#sequential/dense_3/Tensordot/MatMulMatMul-sequential/dense_3/Tensordot/Reshape:output:03sequential/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџo
$sequential/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*sequential/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
%sequential/dense_3/Tensordot/concat_1ConcatV2.sequential/dense_3/Tensordot/GatherV2:output:0-sequential/dense_3/Tensordot/Const_2:output:03sequential/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:О
sequential/dense_3/TensordotReshape-sequential/dense_3/Tensordot/MatMul:product:0.sequential/dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0З
sequential/dense_3/BiasAddBiasAdd%sequential/dense_3/Tensordot:output:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџe
 sequential/activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
sequential/activation/Gelu/mulMul)sequential/activation/Gelu/mul/x:output:0#sequential/dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџf
!sequential/activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?Ж
"sequential/activation/Gelu/truedivRealDiv#sequential/dense_3/BiasAdd:output:0*sequential/activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
sequential/activation/Gelu/ErfErf&sequential/activation/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџe
 sequential/activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ў
sequential/activation/Gelu/addAddV2)sequential/activation/Gelu/add/x:output:0"sequential/activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџЇ
 sequential/activation/Gelu/mul_1Mul"sequential/activation/Gelu/mul:z:0"sequential/activation/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџ
sequential/dropout_1/IdentityIdentity$sequential/activation/Gelu/mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџЁ
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: Ї
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:В
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:С
&sequential/dense_4/Tensordot/transpose	Transpose&sequential/dropout_1/Identity:output:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџУ
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџУ
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Н
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
sequential/dropout_2/IdentityIdentity#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@g
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
valueB:
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
value	B :д
 drop_path/random_uniform_1/shapePack"drop_path/strided_slice_1:output:0+drop_path/random_uniform_1/shape/1:output:0+drop_path/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ї
(drop_path/random_uniform_1/RandomUniformRandomUniform)drop_path/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0V
drop_path/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/add_1AddV2drop_path/add_1/x:output:01drop_path/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџe
drop_path/Floor_1Floordrop_path/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
drop_path/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/truediv_1RealDiv&sequential/dropout_2/Identity:output:0drop_path/truediv_1/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@}
drop_path/mul_1Muldrop_path/truediv_1:z:0drop_path/Floor_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@c
add_1AddV2add:z:0drop_path/mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@д
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp,^sequential/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp^window_attention/Gather*^window_attention/Reshape_1/ReadVariableOp0^window_attention/dense_1/BiasAdd/ReadVariableOp2^window_attention/dense_1/Tensordot/ReadVariableOp0^window_attention/dense_2/BiasAdd/ReadVariableOp2^window_attention/dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : : : 2\
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
:џџџџџџџџџ@

_user_specified_namex
я
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_13292

inputs

identity_1T
IdentityIdentityinputs*
T0*-
_output_shapes
:џџџџџџџџџa

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ

ѕ
C__inference_dense_10_layer_call_and_return_conditional_losses_12718

inputs1
matmul_readvariableop_resource:	W-
biasadd_readvariableop_resource:W
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	W*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџWr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:W*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџWV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџW`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџWw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§d
Х
__inference__wrapped_model_9941
input_1,
model_patch_embedding_9786:@(
model_patch_embedding_9788:@-
model_patch_embedding_9790:	@)
model_swin_transformer_9826:@)
model_swin_transformer_9828:@.
model_swin_transformer_9830:	@Р*
model_swin_transformer_9832:	Р-
model_swin_transformer_9834:	-
model_swin_transformer_9836:	-
model_swin_transformer_9838:@@)
model_swin_transformer_9840:@)
model_swin_transformer_9842:@)
model_swin_transformer_9844:@.
model_swin_transformer_9846:	@*
model_swin_transformer_9848:	.
model_swin_transformer_9850:	@)
model_swin_transformer_9852:@+
model_swin_transformer_1_9890:@+
model_swin_transformer_1_9892:@0
model_swin_transformer_1_9894:	@Р,
model_swin_transformer_1_9896:	Р/
model_swin_transformer_1_9898:	/
model_swin_transformer_1_9900:	4
model_swin_transformer_1_9902:/
model_swin_transformer_1_9904:@@+
model_swin_transformer_1_9906:@+
model_swin_transformer_1_9908:@+
model_swin_transformer_1_9910:@0
model_swin_transformer_1_9912:	@,
model_swin_transformer_1_9914:	0
model_swin_transformer_1_9916:	@+
model_swin_transformer_1_9918:@,
model_patch_merging_9928:
@
-model_dense_10_matmul_readvariableop_resource:	W<
.model_dense_10_biasadd_readvariableop_resource:W
identityЂ%model/dense_10/BiasAdd/ReadVariableOpЂ$model/dense_10/MatMul/ReadVariableOpЂ-model/patch_embedding/StatefulPartitionedCallЂ+model/patch_merging/StatefulPartitionedCallЂ.model/swin_transformer/StatefulPartitionedCallЂ0model/swin_transformer_1/StatefulPartitionedCallN
model/random_crop/ShapeShapeinput_1*
T0*
_output_shapes
:x
%model/random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџz
'model/random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџq
'model/random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
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
ўџџџџџџџџ|
)model/random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџs
)model/random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
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
value	B :@
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
 *  B
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
value	B :@
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
 *  B
model/random_crop/truediv_1RealDivmodel/random_crop/Cast_2:y:0&model/random_crop/truediv_1/y:output:0*
T0*
_output_shapes
: q
model/random_crop/Cast_3Castmodel/random_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 
model/random_crop/MinimumMinimum(model/random_crop/strided_slice:output:0model/random_crop/Cast_1:y:0*
T0*
_output_shapes
: 
model/random_crop/Minimum_1Minimum*model/random_crop/strided_slice_1:output:0model/random_crop/Cast_3:y:0*
T0*
_output_shapes
: 
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
 *   @
model/random_crop/truediv_2RealDivmodel/random_crop/Cast_4:y:0&model/random_crop/truediv_2/y:output:0*
T0*
_output_shapes
: q
model/random_crop/Cast_5Castmodel/random_crop/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: 
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
 *   @
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
value	B : б
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
џџџџџџџџџf
model/random_crop/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџл
model/random_crop/stack_1Pack$model/random_crop/stack_1/0:output:0model/random_crop/Minimum:z:0model/random_crop/Minimum_1:z:0$model/random_crop/stack_1/3:output:0*
N*
T0*
_output_shapes
:П
model/random_crop/SliceSliceinput_1 model/random_crop/stack:output:0"model/random_crop/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"џџџџџџџџџ@@џџџџџџџџџn
model/random_crop/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   з
'model/random_crop/resize/ResizeBilinearResizeBilinear model/random_crop/Slice:output:0&model/random_crop/resize/size:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
half_pixel_centers(н
#model/patch_extract/PartitionedCallPartitionedCall8model/random_crop/resize/ResizeBilinear:resized_images:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9773П
-model/patch_embedding/StatefulPartitionedCallStatefulPartitionedCall,model/patch_extract/PartitionedCall:output:0model_patch_embedding_9786model_patch_embedding_9788model_patch_embedding_9790*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9785Ђ
.model/swin_transformer/StatefulPartitionedCallStatefulPartitionedCall6model/patch_embedding/StatefulPartitionedCall:output:0model_swin_transformer_9826model_swin_transformer_9828model_swin_transformer_9830model_swin_transformer_9832model_swin_transformer_9834model_swin_transformer_9836model_swin_transformer_9838model_swin_transformer_9840model_swin_transformer_9842model_swin_transformer_9844model_swin_transformer_9846model_swin_transformer_9848model_swin_transformer_9850model_swin_transformer_9852*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9825т
0model/swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall7model/swin_transformer/StatefulPartitionedCall:output:0model_swin_transformer_1_9890model_swin_transformer_1_9892model_swin_transformer_1_9894model_swin_transformer_1_9896model_swin_transformer_1_9898model_swin_transformer_1_9900model_swin_transformer_1_9902model_swin_transformer_1_9904model_swin_transformer_1_9906model_swin_transformer_1_9908model_swin_transformer_1_9910model_swin_transformer_1_9912model_swin_transformer_1_9914model_swin_transformer_1_9916model_swin_transformer_1_9918*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9889
+model/patch_merging/StatefulPartitionedCallStatefulPartitionedCall9model/swin_transformer_1/StatefulPartitionedCall:output:0model_patch_merging_9928*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9927w
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :д
#model/global_average_pooling1d/MeanMean4model/patch_merging/StatefulPartitionedCall:output:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
$model/dense_10/MatMul/ReadVariableOpReadVariableOp-model_dense_10_matmul_readvariableop_resource*
_output_shapes
:	W*
dtype0­
model/dense_10/MatMulMatMul,model/global_average_pooling1d/Mean:output:0,model/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџW
%model/dense_10/BiasAdd/ReadVariableOpReadVariableOp.model_dense_10_biasadd_readvariableop_resource*
_output_shapes
:W*
dtype0Ѓ
model/dense_10/BiasAddBiasAddmodel/dense_10/MatMul:product:0-model/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџWt
model/dense_10/SoftmaxSoftmaxmodel/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџWo
IdentityIdentity model/dense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџWз
NoOpNoOp&^model/dense_10/BiasAdd/ReadVariableOp%^model/dense_10/MatMul/ReadVariableOp.^model/patch_embedding/StatefulPartitionedCall,^model/patch_merging/StatefulPartitionedCall/^model/swin_transformer/StatefulPartitionedCall1^model/swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%model/dense_10/BiasAdd/ReadVariableOp%model/dense_10/BiasAdd/ReadVariableOp2L
$model/dense_10/MatMul/ReadVariableOp$model/dense_10/MatMul/ReadVariableOp2^
-model/patch_embedding/StatefulPartitionedCall-model/patch_embedding/StatefulPartitionedCall2Z
+model/patch_merging/StatefulPartitionedCall+model/patch_merging/StatefulPartitionedCall2`
.model/swin_transformer/StatefulPartitionedCall.model/swin_transformer/StatefulPartitionedCall2d
0model/swin_transformer_1/StatefulPartitionedCall0model/swin_transformer_1/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@@
!
_user_specified_name	input_1
ћ"
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
§џџџџџџџџh
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
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
ўџџџџџџџџj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
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
 *  BQ
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
 *  BW
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
џџџџџџџџџT
	stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
stack_1Packstack_1/0:output:0Minimum:z:0Minimum_1:z:0stack_1/3:output:0*
N*
T0*
_output_shapes
:
SliceSliceinputsstack:output:0stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"џџџџџџџџџ@@џџџџџџџџџ\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   Ё
resize/ResizeBilinearResizeBilinearSlice:output:0resize/size:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
half_pixel_centers(v
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:џџџџџџџџџ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
д
њ
B__inference_dense_4_layer_call_and_return_conditional_losses_13194

inputs4
!tensordot_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
X
з
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

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ѕ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*Ю
valueФBС,B9layer_with_weights-2/attn_mask/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEBJlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHХ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_swin_transformer_1_variable_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop7savev2_patch_embedding_dense_kernel_read_readvariableop5savev2_patch_embedding_dense_bias_read_readvariableop?savev2_patch_embedding_embedding_embeddings_read_readvariableopEsavev2_swin_transformer_layer_normalization_gamma_read_readvariableopDsavev2_swin_transformer_layer_normalization_beta_read_readvariableopCsavev2_swin_transformer_window_attention_weight_read_readvariableopKsavev2_swin_transformer_window_attention_dense_1_kernel_read_readvariableopIsavev2_swin_transformer_window_attention_dense_1_bias_read_readvariableopKsavev2_swin_transformer_window_attention_dense_2_kernel_read_readvariableopIsavev2_swin_transformer_window_attention_dense_2_bias_read_readvariableopGsavev2_swin_transformer_layer_normalization_1_gamma_read_readvariableopFsavev2_swin_transformer_layer_normalization_1_beta_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableopEsavev2_swin_transformer_window_attention_variable_read_readvariableopIsavev2_swin_transformer_1_layer_normalization_2_gamma_read_readvariableopHsavev2_swin_transformer_1_layer_normalization_2_beta_read_readvariableopGsavev2_swin_transformer_1_window_attention_1_weight_read_readvariableopOsavev2_swin_transformer_1_window_attention_1_dense_5_kernel_read_readvariableopMsavev2_swin_transformer_1_window_attention_1_dense_5_bias_read_readvariableopOsavev2_swin_transformer_1_window_attention_1_dense_6_kernel_read_readvariableopMsavev2_swin_transformer_1_window_attention_1_dense_6_bias_read_readvariableopIsavev2_swin_transformer_1_layer_normalization_3_gamma_read_readvariableopHsavev2_swin_transformer_1_layer_normalization_3_beta_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableopIsavev2_swin_transformer_1_window_attention_1_variable_read_readvariableop7savev2_patch_merging_dense_9_kernel_read_readvariableop%savev2_statevar_1_read_readvariableop#savev2_statevar_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,				
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*к
_input_shapesШ
Х: ::	W:W:@:@:	@:@:@:	:	@Р:Р:@@:@:@:@:	@::	@:@::@:@:	:	@Р:Р:@@:@:@:@:	@::	@:@::
::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
::%!

_output_shapes
:	W: 

_output_shapes
:W:$ 

_output_shapes

:@: 

_output_shapes
:@:%!

_output_shapes
:	@: 
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
:	@Р:!

_output_shapes	
:Р:$ 

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
:	@:!

_output_shapes	
::%!

_output_shapes
:	@: 
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
:	@Р:!

_output_shapes	
:Р:$ 

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
:	@:!

_output_shapes	
::% !

_output_shapes
:	@: !

_output_shapes
:@:$" 

_output_shapes

::&#"
 
_output_shapes
:
: $
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
Ђ

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_13304

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?j
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Ќ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:џџџџџџџџџu
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:џџџџџџџџџo
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:џџџџџџџџџ_
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


1__inference_swin_transformer_1_layer_call_fn_1176
x
unknown:@
	unknown_0:@
	unknown_1:	@Р
	unknown_2:	Р
	unknown_3:	
	unknown_4:	 
	unknown_5:
	unknown_6:@@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:	@

unknown_11:	

unknown_12:	@

unknown_13:@
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_1156`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:џџџџџџџџџ@: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
п
б
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2631
xG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@M
:window_attention_dense_1_tensordot_readvariableop_resource:	@РG
8window_attention_dense_1_biasadd_readvariableop_resource:	РD
2window_attention_reshape_1_readvariableop_resource:	2
 window_attention_gather_resource:	L
:window_attention_dense_2_tensordot_readvariableop_resource:@@F
8window_attention_dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@G
4sequential_dense_3_tensordot_readvariableop_resource:	@A
2sequential_dense_3_biasadd_readvariableop_resource:	G
4sequential_dense_4_tensordot_readvariableop_resource:	@@
2sequential_dense_4_biasadd_readvariableop_resource:@
identityЂ,layer_normalization/batchnorm/ReadVariableOpЂ0layer_normalization/batchnorm/mul/ReadVariableOpЂ.layer_normalization_1/batchnorm/ReadVariableOpЂ2layer_normalization_1/batchnorm/mul/ReadVariableOpЂ)sequential/dense_3/BiasAdd/ReadVariableOpЂ+sequential/dense_3/Tensordot/ReadVariableOpЂ)sequential/dense_4/BiasAdd/ReadVariableOpЂ+sequential/dense_4/Tensordot/ReadVariableOpЂwindow_attention/GatherЂ)window_attention/Reshape_1/ReadVariableOpЂ/window_attention/dense_1/BiasAdd/ReadVariableOpЂ1window_attention/dense_1/Tensordot/ReadVariableOpЂ/window_attention/dense_2/BiasAdd/ReadVariableOpЂ1window_attention/dense_2/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:А
 layer_normalization/moments/meanMeanx;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЏ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencex1layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ш
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7О
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџІ
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Т
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
#layer_normalization/batchnorm/mul_1Mulx%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Г
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0О
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Г
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   
ReshapeReshape'layer_normalization/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_1ReshapeReshape:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@­
1window_attention/dense_1/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@Р*
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
value	B : 
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
value	B : Ѓ
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
valueB: Й
'window_attention/dense_1/Tensordot/ProdProd4window_attention/dense_1/Tensordot/GatherV2:output:01window_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
)window_attention/dense_1/Tensordot/Prod_1Prod6window_attention/dense_1/Tensordot/GatherV2_1:output:03window_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)window_attention/dense_1/Tensordot/concatConcatV20window_attention/dense_1/Tensordot/free:output:00window_attention/dense_1/Tensordot/axes:output:07window_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
(window_attention/dense_1/Tensordot/stackPack0window_attention/dense_1/Tensordot/Prod:output:02window_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:З
,window_attention/dense_1/Tensordot/transpose	TransposeReshape_3:output:02window_attention/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@е
*window_attention/dense_1/Tensordot/ReshapeReshape0window_attention/dense_1/Tensordot/transpose:y:01window_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџж
)window_attention/dense_1/Tensordot/MatMulMatMul3window_attention/dense_1/Tensordot/Reshape:output:09window_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџРu
*window_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Рr
0window_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention/dense_1/Tensordot/concat_1ConcatV24window_attention/dense_1/Tensordot/GatherV2:output:03window_attention/dense_1/Tensordot/Const_2:output:09window_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Я
"window_attention/dense_1/TensordotReshape3window_attention/dense_1/Tensordot/MatMul:product:04window_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџРЅ
/window_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype0Ш
 window_attention/dense_1/BiasAddBiasAdd+window_attention/dense_1/Tensordot:output:07window_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџР{
window_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Е
window_attention/ReshapeReshape)window_attention/dense_1/BiasAdd:output:0'window_attention/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
window_attention/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                В
window_attention/transpose	Transpose!window_attention/Reshape:output:0(window_attention/transpose/perm:output:0*
T0*3
_output_shapes!
:џџџџџџџџџn
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
valueB:О
window_attention/strided_sliceStridedSlicewindow_attention/transpose:y:0-window_attention/strided_slice/stack:output:0/window_attention/strided_slice/stack_1:output:0/window_attention/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:Ц
 window_attention/strided_slice_1StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_1/stack:output:01window_attention/strided_slice_1/stack_1:output:01window_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:Ц
 window_attention/strided_slice_2StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_2/stack:output:01window_attention/strided_slice_2/stack_1:output:01window_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask[
window_attention/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ>
window_attention/mulMul'window_attention/strided_slice:output:0window_attention/mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџz
!window_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             К
window_attention/transpose_1	Transpose)window_attention/strided_slice_1:output:0*window_attention/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
window_attention/matmulBatchMatMulV2window_attention/mul:z:0 window_attention/transpose_1:y:0*
T0*/
_output_shapes
:џџџџџџџџџ
)window_attention/Reshape_1/ReadVariableOpReadVariableOp2window_attention_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	s
 window_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЈ
window_attention/Reshape_1Reshape1window_attention/Reshape_1/ReadVariableOp:value:0)window_attention/Reshape_1/shape:output:0*
T0	*
_output_shapes
:­
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
valueB"      џџџџЁ
window_attention/Reshape_2Reshape"window_attention/Identity:output:0)window_attention/Reshape_2/shape:output:0*
T0*"
_output_shapes
:v
!window_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ї
window_attention/transpose_2	Transpose#window_attention/Reshape_2:output:0*window_attention/transpose_2/perm:output:0*
T0*"
_output_shapes
:a
window_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : І
window_attention/ExpandDims
ExpandDims window_attention/transpose_2:y:0(window_attention/ExpandDims/dim:output:0*
T0*&
_output_shapes
:
window_attention/addAddV2 window_attention/matmul:output:0$window_attention/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
window_attention/SoftmaxSoftmaxwindow_attention/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџk
&window_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?К
$window_attention/dropout/dropout/MulMul"window_attention/Softmax:softmax:0/window_attention/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџx
&window_attention/dropout/dropout/ShapeShape"window_attention/Softmax:softmax:0*
T0*
_output_shapes
:Ц
=window_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform/window_attention/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0t
/window_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<љ
-window_attention/dropout/dropout/GreaterEqualGreaterEqualFwindow_attention/dropout/dropout/random_uniform/RandomUniform:output:08window_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџЉ
%window_attention/dropout/dropout/CastCast1window_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџМ
&window_attention/dropout/dropout/Mul_1Mul(window_attention/dropout/dropout/Mul:z:0)window_attention/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџЛ
window_attention/matmul_1BatchMatMulV2*window_attention/dropout/dropout/Mul_1:z:0)window_attention/strided_slice_2:output:0*
T0*/
_output_shapes
:џџџџџџџџџz
!window_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Г
window_attention/transpose_3	Transpose"window_attention/matmul_1:output:0*window_attention/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџu
 window_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   Ј
window_attention/Reshape_3Reshape window_attention/transpose_3:y:0)window_attention/Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ќ
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
value	B : 
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
value	B : Ѓ
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
valueB: Й
'window_attention/dense_2/Tensordot/ProdProd4window_attention/dense_2/Tensordot/GatherV2:output:01window_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
)window_attention/dense_2/Tensordot/Prod_1Prod6window_attention/dense_2/Tensordot/GatherV2_1:output:03window_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)window_attention/dense_2/Tensordot/concatConcatV20window_attention/dense_2/Tensordot/free:output:00window_attention/dense_2/Tensordot/axes:output:07window_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
(window_attention/dense_2/Tensordot/stackPack0window_attention/dense_2/Tensordot/Prod:output:02window_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ш
,window_attention/dense_2/Tensordot/transpose	Transpose#window_attention/Reshape_3:output:02window_attention/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@е
*window_attention/dense_2/Tensordot/ReshapeReshape0window_attention/dense_2/Tensordot/transpose:y:01window_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџе
)window_attention/dense_2/Tensordot/MatMulMatMul3window_attention/dense_2/Tensordot/Reshape:output:09window_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@t
*window_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@r
0window_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention/dense_2/Tensordot/concat_1ConcatV24window_attention/dense_2/Tensordot/GatherV2:output:03window_attention/dense_2/Tensordot/Const_2:output:09window_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ю
"window_attention/dense_2/TensordotReshape3window_attention/dense_2/Tensordot/MatMul:product:04window_attention/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Є
/window_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ч
 window_attention/dense_2/BiasAddBiasAdd+window_attention/dense_2/Tensordot:output:07window_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@m
(window_attention/dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?С
&window_attention/dropout/dropout_1/MulMul)window_attention/dense_2/BiasAdd:output:01window_attention/dropout/dropout_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@
(window_attention/dropout/dropout_1/ShapeShape)window_attention/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:Ц
?window_attention/dropout/dropout_1/random_uniform/RandomUniformRandomUniform1window_attention/dropout/dropout_1/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@*
dtype0v
1window_attention/dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<ћ
/window_attention/dropout/dropout_1/GreaterEqualGreaterEqualHwindow_attention/dropout/dropout_1/random_uniform/RandomUniform:output:0:window_attention/dropout/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Љ
'window_attention/dropout/dropout_1/CastCast3window_attention/dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ@О
(window_attention/dropout/dropout_1/Mul_1Mul*window_attention/dropout/dropout_1/Mul:z:0+window_attention/dropout/dropout_1/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   
	Reshape_4Reshape,window_attention/dropout/dropout_1/Mul_1:z:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   y
	Reshape_7ReshapeReshape_6:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Q
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
valueB:
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
value	B :Ь
drop_path/random_uniform/shapePack drop_path/strided_slice:output:0)drop_path/random_uniform/shape/1:output:0)drop_path/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:Ѓ
&drop_path/random_uniform/RandomUniformRandomUniform'drop_path/random_uniform/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0T
drop_path/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/addAddV2drop_path/add/x:output:0/drop_path/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџa
drop_path/FloorFloordrop_path/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
drop_path/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/truedivRealDivReshape_7:output:0drop_path/truediv/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@w
drop_path/mulMuldrop_path/truediv:z:0drop_path/Floor:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@Y
addAddV2xdrop_path/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:К
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЙ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ё
+sequential/dense_3/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: Ї
!sequential/dense_3/Tensordot/ProdProd.sequential/dense_3/Tensordot/GatherV2:output:0+sequential/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_3/Tensordot/Prod_1Prod0sequential/dense_3/Tensordot/GatherV2_1:output:0-sequential/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
#sequential/dense_3/Tensordot/concatConcatV2*sequential/dense_3/Tensordot/free:output:0*sequential/dense_3/Tensordot/axes:output:01sequential/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:В
"sequential/dense_3/Tensordot/stackPack*sequential/dense_3/Tensordot/Prod:output:0,sequential/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:У
&sequential/dense_3/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0,sequential/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@У
$sequential/dense_3/Tensordot/ReshapeReshape*sequential/dense_3/Tensordot/transpose:y:0+sequential/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџФ
#sequential/dense_3/Tensordot/MatMulMatMul-sequential/dense_3/Tensordot/Reshape:output:03sequential/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџo
$sequential/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*sequential/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
%sequential/dense_3/Tensordot/concat_1ConcatV2.sequential/dense_3/Tensordot/GatherV2:output:0-sequential/dense_3/Tensordot/Const_2:output:03sequential/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:О
sequential/dense_3/TensordotReshape-sequential/dense_3/Tensordot/MatMul:product:0.sequential/dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0З
sequential/dense_3/BiasAddBiasAdd%sequential/dense_3/Tensordot:output:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџe
 sequential/activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
sequential/activation/Gelu/mulMul)sequential/activation/Gelu/mul/x:output:0#sequential/dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџf
!sequential/activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?Ж
"sequential/activation/Gelu/truedivRealDiv#sequential/dense_3/BiasAdd:output:0*sequential/activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
sequential/activation/Gelu/ErfErf&sequential/activation/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџe
 sequential/activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ў
sequential/activation/Gelu/addAddV2)sequential/activation/Gelu/add/x:output:0"sequential/activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџЇ
 sequential/activation/Gelu/mul_1Mul"sequential/activation/Gelu/mul:z:0"sequential/activation/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџg
"sequential/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?В
 sequential/dropout_1/dropout/MulMul$sequential/activation/Gelu/mul_1:z:0+sequential/dropout_1/dropout/Const:output:0*
T0*-
_output_shapes
:џџџџџџџџџv
"sequential/dropout_1/dropout/ShapeShape$sequential/activation/Gelu/mul_1:z:0*
T0*
_output_shapes
:М
9sequential/dropout_1/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_1/dropout/Shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ*
dtype0p
+sequential/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<ы
)sequential/dropout_1/dropout/GreaterEqualGreaterEqualBsequential/dropout_1/dropout/random_uniform/RandomUniform:output:04sequential/dropout_1/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
!sequential/dropout_1/dropout/CastCast-sequential/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:џџџџџџџџџЎ
"sequential/dropout_1/dropout/Mul_1Mul$sequential/dropout_1/dropout/Mul:z:0%sequential/dropout_1/dropout/Cast:y:0*
T0*-
_output_shapes
:џџџџџџџџџЁ
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: Ї
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:В
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:С
&sequential/dense_4/Tensordot/transpose	Transpose&sequential/dropout_1/dropout/Mul_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџУ
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџУ
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Н
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@g
"sequential/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?А
 sequential/dropout_2/dropout/MulMul#sequential/dense_4/BiasAdd:output:0+sequential/dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@u
"sequential/dropout_2/dropout/ShapeShape#sequential/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:Л
9sequential/dropout_2/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
dtype0p
+sequential/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<ъ
)sequential/dropout_2/dropout/GreaterEqualGreaterEqualBsequential/dropout_2/dropout/random_uniform/RandomUniform:output:04sequential/dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
!sequential/dropout_2/dropout/CastCast-sequential/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ@­
"sequential/dropout_2/dropout/Mul_1Mul$sequential/dropout_2/dropout/Mul:z:0%sequential/dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@g
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
valueB:
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
value	B :д
 drop_path/random_uniform_1/shapePack"drop_path/strided_slice_1:output:0+drop_path/random_uniform_1/shape/1:output:0+drop_path/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ї
(drop_path/random_uniform_1/RandomUniformRandomUniform)drop_path/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0V
drop_path/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/add_1AddV2drop_path/add_1/x:output:01drop_path/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџe
drop_path/Floor_1Floordrop_path/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
drop_path/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/truediv_1RealDiv&sequential/dropout_2/dropout/Mul_1:z:0drop_path/truediv_1/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@}
drop_path/mul_1Muldrop_path/truediv_1:z:0drop_path/Floor_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@c
add_1AddV2add:z:0drop_path/mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@д
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp,^sequential/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp^window_attention/Gather*^window_attention/Reshape_1/ReadVariableOp0^window_attention/dense_1/BiasAdd/ReadVariableOp2^window_attention/dense_1/Tensordot/ReadVariableOp0^window_attention/dense_2/BiasAdd/ReadVariableOp2^window_attention/dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : : : 2\
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
:џџџџџџџџџ@

_user_specified_namex
я
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_13143

inputs

identity_1T
IdentityIdentityinputs*
T0*-
_output_shapes
:џџџџџџџџџa

Identity_1IdentityIdentity:output:0*
T0*-
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш

F__inference_random_crop_layer_call_and_return_conditional_losses_12000

inputs
cond_input_1:	
identityЂcond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџh
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
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
ўџџџџџџџџj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
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
: Ц
condIfAll:output:0inputscond_input_1*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_11855*.
output_shapes
:џџџџџџџџџ@@*"
then_branchR
cond_true_11854b
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@m
IdentityIdentitycond/Identity:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: 2
condcond:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ѕ
Ж
%__inference_model_layer_call_fn_11206

inputs
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:@
	unknown_3:@
	unknown_4:	@Р
	unknown_5:	Р
	unknown_6:	
	unknown_7:	
	unknown_8:@@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:	@

unknown_13:	

unknown_14:	@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@Р

unknown_19:	Р

unknown_20:	

unknown_21:	!

unknown_22:

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:	@

unknown_28:	

unknown_29:	@

unknown_30:@

unknown_31:


unknown_32:	W

unknown_33:W
identityЂStatefulPartitionedCall
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
:џџџџџџџџџW*E
_read_only_resource_inputs'
%#	
 !"#*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџW`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
С
G
+__inference_random_crop_layer_call_fn_11774

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10007h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
ј
T
8__inference_global_average_pooling1d_layer_call_fn_12692

inputs
identityЦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
е
Е
#__inference_signature_wrapper_11131
input_1
unknown:@
	unknown_0:@
	unknown_1:	@
	unknown_2:@
	unknown_3:@
	unknown_4:	@Р
	unknown_5:	Р
	unknown_6:	
	unknown_7:	
	unknown_8:@@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:	@

unknown_13:	

unknown_14:	@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@Р

unknown_19:	Р

unknown_20:	

unknown_21:	!

unknown_22:

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:	@

unknown_28:	

unknown_29:	@

unknown_30:@

unknown_31:


unknown_32:	W

unknown_33:W
identityЂStatefulPartitionedCallњ
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
:џџџџџџџџџW*E
_read_only_resource_inputs'
%#	
 !"#*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_9941o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџW`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@@
!
_user_specified_name	input_1
­
}
,__inference_patch_merging_layer_call_fn_2744
x
unknown:

identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallxunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_patch_merging_layer_call_and_return_conditional_losses_2738`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:џџџџџџџџџ@: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
,
_output_shapes
:џџџџџџџџџ@

_user_specified_namex
Ѓ

ѕ
C__inference_dense_10_layer_call_and_return_conditional_losses_10098

inputs1
matmul_readvariableop_resource:	W-
biasadd_readvariableop_resource:W
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	W*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџWr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:W*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџWV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџW`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџWw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_12511

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџ@`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Д
ш
5map_while_stateless_random_flip_left_right_true_10271v
rmap_while_stateless_random_flip_left_right_reversev2_map_while_stateless_random_flip_left_right_control_dependency7
3map_while_stateless_random_flip_left_right_identity
9map/while/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:І
4map/while/stateless_random_flip_left_right/ReverseV2	ReverseV2rmap_while_stateless_random_flip_left_right_reversev2_map_while_stateless_random_flip_left_right_control_dependencyBmap/while/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*"
_output_shapes
:@@Ћ
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
Ь
М
random_crop_cond_true_11438!
random_crop_cond_shape_inputsG
9random_crop_cond_stateful_uniform_rngreadandskip_resource:	
random_crop_cond_identityЂ3random_crop/cond/crop_to_bounding_box/Assert/AssertЂ5random_crop/cond/crop_to_bounding_box/Assert_1/AssertЂ5random_crop/cond/crop_to_bounding_box/Assert_2/AssertЂ5random_crop/cond/crop_to_bounding_box/Assert_3/AssertЂ0random_crop/cond/stateful_uniform/RngReadAndSkipc
random_crop/cond/ShapeShaperandom_crop_cond_shape_inputs*
T0*
_output_shapes
:w
$random_crop/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџy
&random_crop/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџp
&random_crop/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
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
value	B :@
random_crop/cond/subSub'random_crop/cond/strided_slice:output:0random_crop/cond/sub/y:output:0*
T0*
_output_shapes
: y
&random_crop/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ{
(random_crop/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџr
(random_crop/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
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
value	B :@
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
valueB :џџџџq
'random_crop/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: Г
&random_crop/cond/stateful_uniform/ProdProd0random_crop/cond/stateful_uniform/shape:output:00random_crop/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: j
(random_crop/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
(random_crop/cond/stateful_uniform/Cast_1Cast/random_crop/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: њ
0random_crop/cond/stateful_uniform/RngReadAndSkipRngReadAndSkip9random_crop_cond_stateful_uniform_rngreadandskip_resource1random_crop/cond/stateful_uniform/Cast/x:output:0,random_crop/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:
5random_crop/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7random_crop/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7random_crop/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
/random_crop/cond/stateful_uniform/strided_sliceStridedSlice8random_crop/cond/stateful_uniform/RngReadAndSkip:value:0>random_crop/cond/stateful_uniform/strided_slice/stack:output:0@random_crop/cond/stateful_uniform/strided_slice/stack_1:output:0@random_crop/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
)random_crop/cond/stateful_uniform/BitcastBitcast8random_crop/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
7random_crop/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
9random_crop/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9random_crop/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ї
1random_crop/cond/stateful_uniform/strided_slice_1StridedSlice8random_crop/cond/stateful_uniform/RngReadAndSkip:value:0@random_crop/cond/stateful_uniform/strided_slice_1/stack:output:0Brandom_crop/cond/stateful_uniform/strided_slice_1/stack_1:output:0Brandom_crop/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ѓ
+random_crop/cond/stateful_uniform/Bitcast_1Bitcast:random_crop/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0g
%random_crop/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :
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
valueB:Й
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
: 
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
valueB:Й
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
: 
random_crop/cond/mod_1FloorMod)random_crop/cond/strided_slice_3:output:0random_crop/cond/add_1:z:0*
T0*
_output_shapes
: x
+random_crop/cond/crop_to_bounding_box/ShapeShaperandom_crop_cond_shape_inputs*
T0*
_output_shapes
:
-random_crop/cond/crop_to_bounding_box/unstackUnpack4random_crop/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numv
4random_crop/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : О
2random_crop/cond/crop_to_bounding_box/GreaterEqualGreaterEqualrandom_crop/cond/mod_1:z:0=random_crop/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
2random_crop/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.
:random_crop/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.м
3random_crop/cond/crop_to_bounding_box/Assert/AssertAssert6random_crop/cond/crop_to_bounding_box/GreaterEqual:z:0Crandom_crop/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 x
6random_crop/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : Р
4random_crop/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualrandom_crop/cond/mod:z:0?random_crop/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: 
4random_crop/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
<random_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.
5random_crop/cond/crop_to_bounding_box/Assert_1/AssertAssert8random_crop/cond/crop_to_bounding_box/GreaterEqual_1:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:04^random_crop/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 m
+random_crop/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value	B :@Ѕ
)random_crop/cond/crop_to_bounding_box/addAddV24random_crop/cond/crop_to_bounding_box/add/x:output:0random_crop/cond/mod_1:z:0*
T0*
_output_shapes
: s
1random_crop/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value	B :@Ш
/random_crop/cond/crop_to_bounding_box/LessEqual	LessEqual-random_crop/cond/crop_to_bounding_box/add:z:0:random_crop/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: 
4random_crop/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
<random_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.
5random_crop/cond/crop_to_bounding_box/Assert_2/AssertAssert3random_crop/cond/crop_to_bounding_box/LessEqual:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:06^random_crop/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 o
-random_crop/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value	B :@Ї
+random_crop/cond/crop_to_bounding_box/add_1AddV26random_crop/cond/crop_to_bounding_box/add_1/x:output:0random_crop/cond/mod:z:0*
T0*
_output_shapes
: u
3random_crop/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B :@Ю
1random_crop/cond/crop_to_bounding_box/LessEqual_1	LessEqual/random_crop/cond/crop_to_bounding_box/add_1:z:0<random_crop/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: 
4random_crop/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
<random_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.
5random_crop/cond/crop_to_bounding_box/Assert_3/AssertAssert5random_crop/cond/crop_to_bounding_box/LessEqual_1:z:0Erandom_crop/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:06^random_crop/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 
8random_crop/cond/crop_to_bounding_box/control_dependencyIdentityrandom_crop_cond_shape_inputs4^random_crop/cond/crop_to_bounding_box/Assert/Assert6^random_crop/cond/crop_to_bounding_box/Assert_1/Assert6^random_crop/cond/crop_to_bounding_box/Assert_2/Assert6^random_crop/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*/
_output_shapes
:џџџџџџџџџ@@o
-random_crop/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : o
-random_crop/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
+random_crop/cond/crop_to_bounding_box/stackPack6random_crop/cond/crop_to_bounding_box/stack/0:output:0random_crop/cond/mod:z:0random_crop/cond/mod_1:z:06random_crop/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:
-random_crop/cond/crop_to_bounding_box/Shape_1ShapeArandom_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:
9random_crop/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;random_crop/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;random_crop/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3random_crop/cond/crop_to_bounding_box/strided_sliceStridedSlice6random_crop/cond/crop_to_bounding_box/Shape_1:output:0Brandom_crop/cond/crop_to_bounding_box/strided_slice/stack:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Drandom_crop/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
-random_crop/cond/crop_to_bounding_box/Shape_2ShapeArandom_crop/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:
;random_crop/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
=random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_crop/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
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
value	B :@е
-random_crop/cond/crop_to_bounding_box/stack_1Pack<random_crop/cond/crop_to_bounding_box/strided_slice:output:08random_crop/cond/crop_to_bounding_box/stack_1/1:output:08random_crop/cond/crop_to_bounding_box/stack_1/2:output:0>random_crop/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:Ќ
+random_crop/cond/crop_to_bounding_box/SliceSliceArandom_crop/cond/crop_to_bounding_box/control_dependency:output:04random_crop/cond/crop_to_bounding_box/stack:output:06random_crop/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ@@­
random_crop/cond/IdentityIdentity4random_crop/cond/crop_to_bounding_box/Slice:output:0^random_crop/cond/NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@ш
random_crop/cond/NoOpNoOp4^random_crop/cond/crop_to_bounding_box/Assert/Assert6^random_crop/cond/crop_to_bounding_box/Assert_1/Assert6^random_crop/cond/crop_to_bounding_box/Assert_2/Assert6^random_crop/cond/crop_to_bounding_box/Assert_3/Assert1^random_crop/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "?
random_crop_cond_identity"random_crop/cond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: 2j
3random_crop/cond/crop_to_bounding_box/Assert/Assert3random_crop/cond/crop_to_bounding_box/Assert/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_1/Assert5random_crop/cond/crop_to_bounding_box/Assert_1/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_2/Assert5random_crop/cond/crop_to_bounding_box/Assert_2/Assert2n
5random_crop/cond/crop_to_bounding_box/Assert_3/Assert5random_crop/cond/crop_to_bounding_box/Assert_3/Assert2d
0random_crop/cond/stateful_uniform/RngReadAndSkip0random_crop/cond/stateful_uniform/RngReadAndSkip:5 1
/
_output_shapes
:џџџџџџџџџ@@
д	

Arandom_flip_map_while_stateless_random_flip_left_right_true_11655
random_flip_map_while_stateless_random_flip_left_right_reversev2_random_flip_map_while_stateless_random_flip_left_right_control_dependencyC
?random_flip_map_while_stateless_random_flip_left_right_identity
Erandom_flip/map/while/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:з
@random_flip/map/while/stateless_random_flip_left_right/ReverseV2	ReverseV2random_flip_map_while_stateless_random_flip_left_right_reversev2_random_flip_map_while_stateless_random_flip_left_right_control_dependencyNrandom_flip/map/while/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*"
_output_shapes
:@@У
?random_flip/map/while/stateless_random_flip_left_right/IdentityIdentityIrandom_flip/map/while/stateless_random_flip_left_right/ReverseV2:output:0*
T0*"
_output_shapes
:@@"
?random_flip_map_while_stateless_random_flip_left_right_identityHrandom_flip/map/while/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:@@:( $
"
_output_shapes
:@@
6
О
@__inference_model_layer_call_and_return_conditional_losses_10967
input_1'
patch_embedding_10890:@#
patch_embedding_10892:@(
patch_embedding_10894:	@$
swin_transformer_10897:@$
swin_transformer_10899:@)
swin_transformer_10901:	@Р%
swin_transformer_10903:	Р(
swin_transformer_10905:	(
swin_transformer_10907:	(
swin_transformer_10909:@@$
swin_transformer_10911:@$
swin_transformer_10913:@$
swin_transformer_10915:@)
swin_transformer_10917:	@%
swin_transformer_10919:	)
swin_transformer_10921:	@$
swin_transformer_10923:@&
swin_transformer_1_10926:@&
swin_transformer_1_10928:@+
swin_transformer_1_10930:	@Р'
swin_transformer_1_10932:	Р*
swin_transformer_1_10934:	*
swin_transformer_1_10936:	/
swin_transformer_1_10938:*
swin_transformer_1_10940:@@&
swin_transformer_1_10942:@&
swin_transformer_1_10944:@&
swin_transformer_1_10946:@+
swin_transformer_1_10948:	@'
swin_transformer_1_10950:	+
swin_transformer_1_10952:	@&
swin_transformer_1_10954:@'
patch_merging_10957:
!
dense_10_10961:	W
dense_10_10963:W
identityЂ dense_10/StatefulPartitionedCallЂ'patch_embedding/StatefulPartitionedCallЂ%patch_merging/StatefulPartitionedCallЂ(swin_transformer/StatefulPartitionedCallЂ*swin_transformer_1/StatefulPartitionedCallЦ
random_crop/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10007у
random_flip/PartitionedCallPartitionedCall$random_crop/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_10013У
patch_extract/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9773Є
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_10890patch_embedding_10892patch_embedding_10894*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9785а
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_10897swin_transformer_10899swin_transformer_10901swin_transformer_10903swin_transformer_10905swin_transformer_10907swin_transformer_10909swin_transformer_10911swin_transformer_10913swin_transformer_10915swin_transformer_10917swin_transformer_10919swin_transformer_10921swin_transformer_10923*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9825
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_10926swin_transformer_1_10928swin_transformer_1_10930swin_transformer_1_10932swin_transformer_1_10934swin_transformer_1_10936swin_transformer_1_10938swin_transformer_1_10940swin_transformer_1_10942swin_transformer_1_10944swin_transformer_1_10946swin_transformer_1_10948swin_transformer_1_10950swin_transformer_1_10952swin_transformer_1_10954*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9889ќ
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_10957*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9927џ
(global_average_pooling1d/PartitionedCallPartitionedCall.patch_merging/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_9951
 dense_10/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_10_10961dense_10_10963*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџW*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_10098x
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџW
NoOpNoOp!^dense_10/StatefulPartitionedCall(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2R
'patch_embedding/StatefulPartitionedCall'patch_embedding/StatefulPartitionedCall2N
%patch_merging/StatefulPartitionedCall%patch_merging/StatefulPartitionedCall2T
(swin_transformer/StatefulPartitionedCall(swin_transformer/StatefulPartitionedCall2X
*swin_transformer_1/StatefulPartitionedCall*swin_transformer_1/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@@
!
_user_specified_name	input_1
и
ћ
B__inference_dense_7_layer_call_and_return_conditional_losses_12443

inputs4
!tensordot_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџe
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ф
з
*__inference_sequential_layer_call_fn_12244
dense_3_input
unknown:	@
	unknown_0:	
	unknown_1:	@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_3_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_12233t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ@
'
_user_specified_namedense_3_input
ж

'__inference_dense_3_layer_call_fn_13081

inputs
unknown:	@
	unknown_0:	
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_12162u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ћ"
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
§џџџџџџџџh
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
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
ўџџџџџџџџj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
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
 *  BQ
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
 *  BW
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
џџџџџџџџџT
	stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
stack_1Packstack_1/0:output:0Minimum:z:0Minimum_1:z:0stack_1/3:output:0*
N*
T0*
_output_shapes
:
SliceSliceinputsstack:output:0stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"џџџџџџџџџ@@џџџџџџџџџ\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   Ё
resize/ResizeBilinearResizeBilinearSlice:output:0resize/size:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
half_pixel_centers(v
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:џџџџџџџџџ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
п
б
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2087
xG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@M
:window_attention_dense_1_tensordot_readvariableop_resource:	@РG
8window_attention_dense_1_biasadd_readvariableop_resource:	РD
2window_attention_reshape_1_readvariableop_resource:	2
 window_attention_gather_resource:	L
:window_attention_dense_2_tensordot_readvariableop_resource:@@F
8window_attention_dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@G
4sequential_dense_3_tensordot_readvariableop_resource:	@A
2sequential_dense_3_biasadd_readvariableop_resource:	G
4sequential_dense_4_tensordot_readvariableop_resource:	@@
2sequential_dense_4_biasadd_readvariableop_resource:@
identityЂ,layer_normalization/batchnorm/ReadVariableOpЂ0layer_normalization/batchnorm/mul/ReadVariableOpЂ.layer_normalization_1/batchnorm/ReadVariableOpЂ2layer_normalization_1/batchnorm/mul/ReadVariableOpЂ)sequential/dense_3/BiasAdd/ReadVariableOpЂ+sequential/dense_3/Tensordot/ReadVariableOpЂ)sequential/dense_4/BiasAdd/ReadVariableOpЂ+sequential/dense_4/Tensordot/ReadVariableOpЂwindow_attention/GatherЂ)window_attention/Reshape_1/ReadVariableOpЂ/window_attention/dense_1/BiasAdd/ReadVariableOpЂ1window_attention/dense_1/Tensordot/ReadVariableOpЂ/window_attention/dense_2/BiasAdd/ReadVariableOpЂ1window_attention/dense_2/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:А
 layer_normalization/moments/meanMeanx;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЏ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencex1layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ш
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7О
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџІ
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Т
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
#layer_normalization/batchnorm/mul_1Mulx%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Г
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0О
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Г
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   
ReshapeReshape'layer_normalization/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_1ReshapeReshape:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@­
1window_attention/dense_1/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@Р*
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
value	B : 
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
value	B : Ѓ
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
valueB: Й
'window_attention/dense_1/Tensordot/ProdProd4window_attention/dense_1/Tensordot/GatherV2:output:01window_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
)window_attention/dense_1/Tensordot/Prod_1Prod6window_attention/dense_1/Tensordot/GatherV2_1:output:03window_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)window_attention/dense_1/Tensordot/concatConcatV20window_attention/dense_1/Tensordot/free:output:00window_attention/dense_1/Tensordot/axes:output:07window_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
(window_attention/dense_1/Tensordot/stackPack0window_attention/dense_1/Tensordot/Prod:output:02window_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:З
,window_attention/dense_1/Tensordot/transpose	TransposeReshape_3:output:02window_attention/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@е
*window_attention/dense_1/Tensordot/ReshapeReshape0window_attention/dense_1/Tensordot/transpose:y:01window_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџж
)window_attention/dense_1/Tensordot/MatMulMatMul3window_attention/dense_1/Tensordot/Reshape:output:09window_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџРu
*window_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Рr
0window_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention/dense_1/Tensordot/concat_1ConcatV24window_attention/dense_1/Tensordot/GatherV2:output:03window_attention/dense_1/Tensordot/Const_2:output:09window_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Я
"window_attention/dense_1/TensordotReshape3window_attention/dense_1/Tensordot/MatMul:product:04window_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџРЅ
/window_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype0Ш
 window_attention/dense_1/BiasAddBiasAdd+window_attention/dense_1/Tensordot:output:07window_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџР{
window_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Е
window_attention/ReshapeReshape)window_attention/dense_1/BiasAdd:output:0'window_attention/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
window_attention/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                В
window_attention/transpose	Transpose!window_attention/Reshape:output:0(window_attention/transpose/perm:output:0*
T0*3
_output_shapes!
:џџџџџџџџџn
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
valueB:О
window_attention/strided_sliceStridedSlicewindow_attention/transpose:y:0-window_attention/strided_slice/stack:output:0/window_attention/strided_slice/stack_1:output:0/window_attention/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:Ц
 window_attention/strided_slice_1StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_1/stack:output:01window_attention/strided_slice_1/stack_1:output:01window_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:Ц
 window_attention/strided_slice_2StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_2/stack:output:01window_attention/strided_slice_2/stack_1:output:01window_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask[
window_attention/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ>
window_attention/mulMul'window_attention/strided_slice:output:0window_attention/mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџz
!window_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             К
window_attention/transpose_1	Transpose)window_attention/strided_slice_1:output:0*window_attention/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
window_attention/matmulBatchMatMulV2window_attention/mul:z:0 window_attention/transpose_1:y:0*
T0*/
_output_shapes
:џџџџџџџџџ
)window_attention/Reshape_1/ReadVariableOpReadVariableOp2window_attention_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	s
 window_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЈ
window_attention/Reshape_1Reshape1window_attention/Reshape_1/ReadVariableOp:value:0)window_attention/Reshape_1/shape:output:0*
T0	*
_output_shapes
:­
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
valueB"      џџџџЁ
window_attention/Reshape_2Reshape"window_attention/Identity:output:0)window_attention/Reshape_2/shape:output:0*
T0*"
_output_shapes
:v
!window_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ї
window_attention/transpose_2	Transpose#window_attention/Reshape_2:output:0*window_attention/transpose_2/perm:output:0*
T0*"
_output_shapes
:a
window_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : І
window_attention/ExpandDims
ExpandDims window_attention/transpose_2:y:0(window_attention/ExpandDims/dim:output:0*
T0*&
_output_shapes
:
window_attention/addAddV2 window_attention/matmul:output:0$window_attention/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
window_attention/SoftmaxSoftmaxwindow_attention/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџk
&window_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?К
$window_attention/dropout/dropout/MulMul"window_attention/Softmax:softmax:0/window_attention/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџx
&window_attention/dropout/dropout/ShapeShape"window_attention/Softmax:softmax:0*
T0*
_output_shapes
:Ц
=window_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform/window_attention/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0t
/window_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<љ
-window_attention/dropout/dropout/GreaterEqualGreaterEqualFwindow_attention/dropout/dropout/random_uniform/RandomUniform:output:08window_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџЉ
%window_attention/dropout/dropout/CastCast1window_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџМ
&window_attention/dropout/dropout/Mul_1Mul(window_attention/dropout/dropout/Mul:z:0)window_attention/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџЛ
window_attention/matmul_1BatchMatMulV2*window_attention/dropout/dropout/Mul_1:z:0)window_attention/strided_slice_2:output:0*
T0*/
_output_shapes
:џџџџџџџџџz
!window_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Г
window_attention/transpose_3	Transpose"window_attention/matmul_1:output:0*window_attention/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџu
 window_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   Ј
window_attention/Reshape_3Reshape window_attention/transpose_3:y:0)window_attention/Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ќ
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
value	B : 
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
value	B : Ѓ
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
valueB: Й
'window_attention/dense_2/Tensordot/ProdProd4window_attention/dense_2/Tensordot/GatherV2:output:01window_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
)window_attention/dense_2/Tensordot/Prod_1Prod6window_attention/dense_2/Tensordot/GatherV2_1:output:03window_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)window_attention/dense_2/Tensordot/concatConcatV20window_attention/dense_2/Tensordot/free:output:00window_attention/dense_2/Tensordot/axes:output:07window_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
(window_attention/dense_2/Tensordot/stackPack0window_attention/dense_2/Tensordot/Prod:output:02window_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ш
,window_attention/dense_2/Tensordot/transpose	Transpose#window_attention/Reshape_3:output:02window_attention/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@е
*window_attention/dense_2/Tensordot/ReshapeReshape0window_attention/dense_2/Tensordot/transpose:y:01window_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџе
)window_attention/dense_2/Tensordot/MatMulMatMul3window_attention/dense_2/Tensordot/Reshape:output:09window_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@t
*window_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@r
0window_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention/dense_2/Tensordot/concat_1ConcatV24window_attention/dense_2/Tensordot/GatherV2:output:03window_attention/dense_2/Tensordot/Const_2:output:09window_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ю
"window_attention/dense_2/TensordotReshape3window_attention/dense_2/Tensordot/MatMul:product:04window_attention/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Є
/window_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ч
 window_attention/dense_2/BiasAddBiasAdd+window_attention/dense_2/Tensordot:output:07window_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@m
(window_attention/dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?С
&window_attention/dropout/dropout_1/MulMul)window_attention/dense_2/BiasAdd:output:01window_attention/dropout/dropout_1/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@
(window_attention/dropout/dropout_1/ShapeShape)window_attention/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:Ц
?window_attention/dropout/dropout_1/random_uniform/RandomUniformRandomUniform1window_attention/dropout/dropout_1/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@*
dtype0v
1window_attention/dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<ћ
/window_attention/dropout/dropout_1/GreaterEqualGreaterEqualHwindow_attention/dropout/dropout_1/random_uniform/RandomUniform:output:0:window_attention/dropout/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Љ
'window_attention/dropout/dropout_1/CastCast3window_attention/dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ@О
(window_attention/dropout/dropout_1/Mul_1Mul*window_attention/dropout/dropout_1/Mul:z:0+window_attention/dropout/dropout_1/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   
	Reshape_4Reshape,window_attention/dropout/dropout_1/Mul_1:z:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   y
	Reshape_7ReshapeReshape_6:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Q
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
valueB:
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
value	B :Ь
drop_path/random_uniform/shapePack drop_path/strided_slice:output:0)drop_path/random_uniform/shape/1:output:0)drop_path/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:Ѓ
&drop_path/random_uniform/RandomUniformRandomUniform'drop_path/random_uniform/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0T
drop_path/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/addAddV2drop_path/add/x:output:0/drop_path/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџa
drop_path/FloorFloordrop_path/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
drop_path/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/truedivRealDivReshape_7:output:0drop_path/truediv/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@w
drop_path/mulMuldrop_path/truediv:z:0drop_path/Floor:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@Y
addAddV2xdrop_path/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:К
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЙ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ё
+sequential/dense_3/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: Ї
!sequential/dense_3/Tensordot/ProdProd.sequential/dense_3/Tensordot/GatherV2:output:0+sequential/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_3/Tensordot/Prod_1Prod0sequential/dense_3/Tensordot/GatherV2_1:output:0-sequential/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
#sequential/dense_3/Tensordot/concatConcatV2*sequential/dense_3/Tensordot/free:output:0*sequential/dense_3/Tensordot/axes:output:01sequential/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:В
"sequential/dense_3/Tensordot/stackPack*sequential/dense_3/Tensordot/Prod:output:0,sequential/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:У
&sequential/dense_3/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0,sequential/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@У
$sequential/dense_3/Tensordot/ReshapeReshape*sequential/dense_3/Tensordot/transpose:y:0+sequential/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџФ
#sequential/dense_3/Tensordot/MatMulMatMul-sequential/dense_3/Tensordot/Reshape:output:03sequential/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџo
$sequential/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*sequential/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
%sequential/dense_3/Tensordot/concat_1ConcatV2.sequential/dense_3/Tensordot/GatherV2:output:0-sequential/dense_3/Tensordot/Const_2:output:03sequential/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:О
sequential/dense_3/TensordotReshape-sequential/dense_3/Tensordot/MatMul:product:0.sequential/dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0З
sequential/dense_3/BiasAddBiasAdd%sequential/dense_3/Tensordot:output:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџe
 sequential/activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
sequential/activation/Gelu/mulMul)sequential/activation/Gelu/mul/x:output:0#sequential/dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџf
!sequential/activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?Ж
"sequential/activation/Gelu/truedivRealDiv#sequential/dense_3/BiasAdd:output:0*sequential/activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
sequential/activation/Gelu/ErfErf&sequential/activation/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџe
 sequential/activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ў
sequential/activation/Gelu/addAddV2)sequential/activation/Gelu/add/x:output:0"sequential/activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџЇ
 sequential/activation/Gelu/mul_1Mul"sequential/activation/Gelu/mul:z:0"sequential/activation/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџg
"sequential/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?В
 sequential/dropout_1/dropout/MulMul$sequential/activation/Gelu/mul_1:z:0+sequential/dropout_1/dropout/Const:output:0*
T0*-
_output_shapes
:џџџџџџџџџv
"sequential/dropout_1/dropout/ShapeShape$sequential/activation/Gelu/mul_1:z:0*
T0*
_output_shapes
:М
9sequential/dropout_1/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_1/dropout/Shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ*
dtype0p
+sequential/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<ы
)sequential/dropout_1/dropout/GreaterEqualGreaterEqualBsequential/dropout_1/dropout/random_uniform/RandomUniform:output:04sequential/dropout_1/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
!sequential/dropout_1/dropout/CastCast-sequential/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:џџџџџџџџџЎ
"sequential/dropout_1/dropout/Mul_1Mul$sequential/dropout_1/dropout/Mul:z:0%sequential/dropout_1/dropout/Cast:y:0*
T0*-
_output_shapes
:џџџџџџџџџЁ
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: Ї
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:В
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:С
&sequential/dense_4/Tensordot/transpose	Transpose&sequential/dropout_1/dropout/Mul_1:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџУ
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџУ
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Н
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@g
"sequential/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?А
 sequential/dropout_2/dropout/MulMul#sequential/dense_4/BiasAdd:output:0+sequential/dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@u
"sequential/dropout_2/dropout/ShapeShape#sequential/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:Л
9sequential/dropout_2/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
dtype0p
+sequential/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<ъ
)sequential/dropout_2/dropout/GreaterEqualGreaterEqualBsequential/dropout_2/dropout/random_uniform/RandomUniform:output:04sequential/dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
!sequential/dropout_2/dropout/CastCast-sequential/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ@­
"sequential/dropout_2/dropout/Mul_1Mul$sequential/dropout_2/dropout/Mul:z:0%sequential/dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@g
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
valueB:
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
value	B :д
 drop_path/random_uniform_1/shapePack"drop_path/strided_slice_1:output:0+drop_path/random_uniform_1/shape/1:output:0+drop_path/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ї
(drop_path/random_uniform_1/RandomUniformRandomUniform)drop_path/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0V
drop_path/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/add_1AddV2drop_path/add_1/x:output:01drop_path/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџe
drop_path/Floor_1Floordrop_path/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
drop_path/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/truediv_1RealDiv&sequential/dropout_2/dropout/Mul_1:z:0drop_path/truediv_1/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@}
drop_path/mul_1Muldrop_path/truediv_1:z:0drop_path/Floor_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@c
add_1AddV2add:z:0drop_path/mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@д
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp,^sequential/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp^window_attention/Gather*^window_attention/Reshape_1/ReadVariableOp0^window_attention/dense_1/BiasAdd/ReadVariableOp2^window_attention/dense_1/Tensordot/ReadVariableOp0^window_attention/dense_2/BiasAdd/ReadVariableOp2^window_attention/dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : : : 2\
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
:џџџџџџџџџ@

_user_specified_namex
нi

@__inference_model_layer_call_and_return_conditional_losses_11769

inputs&
random_crop_cond_input_1:	+
random_flip_map_while_input_6:	'
patch_embedding_11689:@#
patch_embedding_11691:@(
patch_embedding_11693:	@$
swin_transformer_11696:@$
swin_transformer_11698:@)
swin_transformer_11700:	@Р%
swin_transformer_11702:	Р(
swin_transformer_11704:	(
swin_transformer_11706:	(
swin_transformer_11708:@@$
swin_transformer_11710:@$
swin_transformer_11712:@$
swin_transformer_11714:@)
swin_transformer_11716:	@%
swin_transformer_11718:	)
swin_transformer_11720:	@$
swin_transformer_11722:@&
swin_transformer_1_11725:@&
swin_transformer_1_11727:@+
swin_transformer_1_11729:	@Р'
swin_transformer_1_11731:	Р*
swin_transformer_1_11733:	*
swin_transformer_1_11735:	/
swin_transformer_1_11737:*
swin_transformer_1_11739:@@&
swin_transformer_1_11741:@&
swin_transformer_1_11743:@&
swin_transformer_1_11745:@+
swin_transformer_1_11747:	@'
swin_transformer_1_11749:	+
swin_transformer_1_11751:	@&
swin_transformer_1_11753:@'
patch_merging_11756:
:
'dense_10_matmul_readvariableop_resource:	W6
(dense_10_biasadd_readvariableop_resource:W
identityЂdense_10/BiasAdd/ReadVariableOpЂdense_10/MatMul/ReadVariableOpЂ'patch_embedding/StatefulPartitionedCallЂ%patch_merging/StatefulPartitionedCallЂrandom_crop/condЂrandom_flip/map/whileЂ(swin_transformer/StatefulPartitionedCallЂ*swin_transformer_1/StatefulPartitionedCallG
random_crop/ShapeShapeinputs*
T0*
_output_shapes
:r
random_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§џџџџџџџџt
!random_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџk
!random_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
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
ўџџџџџџџџv
#random_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџm
#random_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
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
value	B : 
random_crop/GreaterEqualGreaterEqualrandom_crop/sub:z:0#random_crop/GreaterEqual/y:output:0*
T0*
_output_shapes
: ^
random_crop/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
random_crop/GreaterEqual_1GreaterEqualrandom_crop/sub_1:z:0%random_crop/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: 
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
value	B :
random_crop/rangeRange random_crop/range/start:output:0random_crop/Rank:output:0 random_crop/range/delta:output:0*
_output_shapes
:
random_crop/All/inputPackrandom_crop/GreaterEqual:z:0random_crop/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:j
random_crop/AllAllrandom_crop/All/input:output:0random_crop/range:output:0*
_output_shapes
: 
random_crop/condIfrandom_crop/All:output:0inputsrandom_crop_cond_input_1*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 */
else_branch R
random_crop_cond_false_11439*.
output_shapes
:џџџџџџџџџ@@*.
then_branchR
random_crop_cond_true_11438z
random_crop/cond/IdentityIdentityrandom_crop/cond:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@g
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
valueB:Ё
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
џџџџџџџџџт
random_flip/map/TensorArrayV2TensorListReserve4random_flip/map/TensorArrayV2/element_shape:output:0&random_flip/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Erandom_flip/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      
7random_flip/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"random_crop/cond/Identity:output:0Nrandom_flip/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвW
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
џџџџџџџџџц
random_flip/map/TensorArrayV2_1TensorListReserve6random_flip/map/TensorArrayV2_1/element_shape:output:0&random_flip/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвd
"random_flip/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : џ
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
: : : : : : : 
@random_flip/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      і
2random_flip/map/TensorArrayV2Stack/TensorListStackTensorListStackrandom_flip/map/while:output:3Irandom_flip/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*/
_output_shapes
:џџџџџџџџџ@@*
element_dtype0к
patch_extract/PartitionedCallPartitionedCall;random_flip/map/TensorArrayV2Stack/TensorListStack:tensor:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9773Є
'patch_embedding/StatefulPartitionedCallStatefulPartitionedCall&patch_extract/PartitionedCall:output:0patch_embedding_11689patch_embedding_11691patch_embedding_11693*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9785б
(swin_transformer/StatefulPartitionedCallStatefulPartitionedCall0patch_embedding/StatefulPartitionedCall:output:0swin_transformer_11696swin_transformer_11698swin_transformer_11700swin_transformer_11702swin_transformer_11704swin_transformer_11706swin_transformer_11708swin_transformer_11710swin_transformer_11712swin_transformer_11714swin_transformer_11716swin_transformer_11718swin_transformer_11720swin_transformer_11722*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_restored_function_body_10622
*swin_transformer_1/StatefulPartitionedCallStatefulPartitionedCall1swin_transformer/StatefulPartitionedCall:output:0swin_transformer_1_11725swin_transformer_1_11727swin_transformer_1_11729swin_transformer_1_11731swin_transformer_1_11733swin_transformer_1_11735swin_transformer_1_11737swin_transformer_1_11739swin_transformer_1_11741swin_transformer_1_11743swin_transformer_1_11745swin_transformer_1_11747swin_transformer_1_11749swin_transformer_1_11751swin_transformer_1_11753*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_restored_function_body_10686ќ
%patch_merging/StatefulPartitionedCallStatefulPartitionedCall3swin_transformer_1/StatefulPartitionedCall:output:0patch_merging_11756*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_restored_function_body_9927q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Т
global_average_pooling1d/MeanMean.patch_merging/StatefulPartitionedCall:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	W*
dtype0
dense_10/MatMulMatMul&global_average_pooling1d/Mean:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџW
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:W*
dtype0
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџWh
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџWi
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџWо
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp(^patch_embedding/StatefulPartitionedCall&^patch_merging/StatefulPartitionedCall^random_crop/cond^random_flip/map/while)^swin_transformer/StatefulPartitionedCall+^swin_transformer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
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
:џџџџџџџџџ@@
 
_user_specified_nameinputs


c
D__inference_dropout_2_layer_call_and_return_conditional_losses_12264

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Ћ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ@n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
З
Э
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
ф
Б
'__inference_restored_function_body_9785	
patch
unknown:@
	unknown_0:@
	unknown_1:	@
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallpatchunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*,
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_patch_embedding_layer_call_and_return_conditional_losses_2689t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:џџџџџџџџџ

_user_specified_namepatch


F__inference_random_flip_layer_call_and_return_conditional_losses_10305

inputs
map_while_input_6:	
identityЂ	map/while?
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
valueB:х
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
џџџџџџџџџО
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      с
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputsBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвK
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
џџџџџџџџџТ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвX
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
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
: : : : : : : 
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"@   @      в
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*/
_output_shapes
:џџџџџџџџџ@@*
element_dtype0
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@R
NoOpNoOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: 2
	map/while	map/while:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

b
F__inference_random_flip_layer_call_and_return_conditional_losses_10013

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
К
Љ
G__inference_sequential_1_layer_call_and_return_conditional_losses_12514

inputs 
dense_7_12444:	@
dense_7_12446:	 
dense_8_12501:	@
dense_8_12503:@
identityЂdense_7/StatefulPartitionedCallЂdense_8/StatefulPartitionedCallя
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_12444dense_7_12446*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_12443ч
activation_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_12461о
dropout_4/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12468
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_8_12501dense_8_12503*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_12500р
dropout_5/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12511v
IdentityIdentity"dropout_5/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ў
ё
G__inference_sequential_1_layer_call_and_return_conditional_losses_12629

inputs 
dense_7_12615:	@
dense_7_12617:	 
dense_8_12622:	@
dense_8_12624:@
identityЂdense_7/StatefulPartitionedCallЂdense_8/StatefulPartitionedCallЂ!dropout_4/StatefulPartitionedCallЂ!dropout_5/StatefulPartitionedCallя
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_12615dense_7_12617*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_12443ч
activation_1/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_12461ю
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12578
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_8_12622dense_8_12624*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_12500
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12545~
IdentityIdentity*dropout_5/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@в
NoOpNoOp ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Е
E
)__inference_dropout_1_layer_call_fn_13133

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12187f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
Э
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

b
)__inference_dropout_1_layer_call_fn_13138

inputs
identityЂStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12297u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Б
E
)__inference_dropout_5_layer_call_fn_13348

inputs
identityД
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_12511e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ@:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
У

(__inference_dense_10_layer_call_fn_12707

inputs
unknown:	W
	unknown_0:W
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџW*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_10098o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџW`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ

Brandom_flip_map_while_stateless_random_flip_left_right_false_11656
random_flip_map_while_stateless_random_flip_left_right_identity_random_flip_map_while_stateless_random_flip_left_right_control_dependencyC
?random_flip_map_while_stateless_random_flip_left_right_identity
?random_flip/map/while/stateless_random_flip_left_right/IdentityIdentityrandom_flip_map_while_stateless_random_flip_left_right_identity_random_flip_map_while_stateless_random_flip_left_right_control_dependency*
T0*"
_output_shapes
:@@"
?random_flip_map_while_stateless_random_flip_left_right_identityHrandom_flip/map/while/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:@@:( $
"
_output_shapes
:@@
З
F
*__inference_activation_layer_call_fn_13116

inputs
identityЖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12180f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К
а
I__inference_swin_transformer_layer_call_and_return_conditional_losses_721
xG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@M
:window_attention_dense_1_tensordot_readvariableop_resource:	@РG
8window_attention_dense_1_biasadd_readvariableop_resource:	РD
2window_attention_reshape_1_readvariableop_resource:	2
 window_attention_gather_resource:	L
:window_attention_dense_2_tensordot_readvariableop_resource:@@F
8window_attention_dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@G
4sequential_dense_3_tensordot_readvariableop_resource:	@A
2sequential_dense_3_biasadd_readvariableop_resource:	G
4sequential_dense_4_tensordot_readvariableop_resource:	@@
2sequential_dense_4_biasadd_readvariableop_resource:@
identityЂ,layer_normalization/batchnorm/ReadVariableOpЂ0layer_normalization/batchnorm/mul/ReadVariableOpЂ.layer_normalization_1/batchnorm/ReadVariableOpЂ2layer_normalization_1/batchnorm/mul/ReadVariableOpЂ)sequential/dense_3/BiasAdd/ReadVariableOpЂ+sequential/dense_3/Tensordot/ReadVariableOpЂ)sequential/dense_4/BiasAdd/ReadVariableOpЂ+sequential/dense_4/Tensordot/ReadVariableOpЂwindow_attention/GatherЂ)window_attention/Reshape_1/ReadVariableOpЂ/window_attention/dense_1/BiasAdd/ReadVariableOpЂ1window_attention/dense_1/Tensordot/ReadVariableOpЂ/window_attention/dense_2/BiasAdd/ReadVariableOpЂ1window_attention/dense_2/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:А
 layer_normalization/moments/meanMeanx;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЏ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencex1layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ш
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7О
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџІ
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Т
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
#layer_normalization/batchnorm/mul_1Mulx%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Г
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0О
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Г
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   
ReshapeReshape'layer_normalization/batchnorm/add_1:z:0Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @p
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_1ReshapeReshape:output:0Reshape_1/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@o
transpose/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   w
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@d
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   x
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@­
1window_attention/dense_1/Tensordot/ReadVariableOpReadVariableOp:window_attention_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@Р*
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
value	B : 
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
value	B : Ѓ
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
valueB: Й
'window_attention/dense_1/Tensordot/ProdProd4window_attention/dense_1/Tensordot/GatherV2:output:01window_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
)window_attention/dense_1/Tensordot/Prod_1Prod6window_attention/dense_1/Tensordot/GatherV2_1:output:03window_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)window_attention/dense_1/Tensordot/concatConcatV20window_attention/dense_1/Tensordot/free:output:00window_attention/dense_1/Tensordot/axes:output:07window_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
(window_attention/dense_1/Tensordot/stackPack0window_attention/dense_1/Tensordot/Prod:output:02window_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:З
,window_attention/dense_1/Tensordot/transpose	TransposeReshape_3:output:02window_attention/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@е
*window_attention/dense_1/Tensordot/ReshapeReshape0window_attention/dense_1/Tensordot/transpose:y:01window_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџж
)window_attention/dense_1/Tensordot/MatMulMatMul3window_attention/dense_1/Tensordot/Reshape:output:09window_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџРu
*window_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Рr
0window_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention/dense_1/Tensordot/concat_1ConcatV24window_attention/dense_1/Tensordot/GatherV2:output:03window_attention/dense_1/Tensordot/Const_2:output:09window_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Я
"window_attention/dense_1/TensordotReshape3window_attention/dense_1/Tensordot/MatMul:product:04window_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџРЅ
/window_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype0Ш
 window_attention/dense_1/BiasAddBiasAdd+window_attention/dense_1/Tensordot:output:07window_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџР{
window_attention/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"џџџџ            Е
window_attention/ReshapeReshape)window_attention/dense_1/BiasAdd:output:0'window_attention/Reshape/shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ|
window_attention/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                В
window_attention/transpose	Transpose!window_attention/Reshape:output:0(window_attention/transpose/perm:output:0*
T0*3
_output_shapes!
:џџџџџџџџџn
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
valueB:О
window_attention/strided_sliceStridedSlicewindow_attention/transpose:y:0-window_attention/strided_slice/stack:output:0/window_attention/strided_slice/stack_1:output:0/window_attention/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:Ц
 window_attention/strided_slice_1StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_1/stack:output:01window_attention/strided_slice_1/stack_1:output:01window_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
valueB:Ц
 window_attention/strided_slice_2StridedSlicewindow_attention/transpose:y:0/window_attention/strided_slice_2/stack:output:01window_attention/strided_slice_2/stack_1:output:01window_attention/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask[
window_attention/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ>
window_attention/mulMul'window_attention/strided_slice:output:0window_attention/mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџz
!window_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             К
window_attention/transpose_1	Transpose)window_attention/strided_slice_1:output:0*window_attention/transpose_1/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
window_attention/matmulBatchMatMulV2window_attention/mul:z:0 window_attention/transpose_1:y:0*
T0*/
_output_shapes
:џџџџџџџџџ
)window_attention/Reshape_1/ReadVariableOpReadVariableOp2window_attention_reshape_1_readvariableop_resource*
_output_shapes

:*
dtype0	s
 window_attention/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЈ
window_attention/Reshape_1Reshape1window_attention/Reshape_1/ReadVariableOp:value:0)window_attention/Reshape_1/shape:output:0*
T0	*
_output_shapes
:­
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
valueB"      џџџџЁ
window_attention/Reshape_2Reshape"window_attention/Identity:output:0)window_attention/Reshape_2/shape:output:0*
T0*"
_output_shapes
:v
!window_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ї
window_attention/transpose_2	Transpose#window_attention/Reshape_2:output:0*window_attention/transpose_2/perm:output:0*
T0*"
_output_shapes
:a
window_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : І
window_attention/ExpandDims
ExpandDims window_attention/transpose_2:y:0(window_attention/ExpandDims/dim:output:0*
T0*&
_output_shapes
:
window_attention/addAddV2 window_attention/matmul:output:0$window_attention/ExpandDims:output:0*
T0*/
_output_shapes
:џџџџџџџџџw
window_attention/SoftmaxSoftmaxwindow_attention/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ
!window_attention/dropout/IdentityIdentity"window_attention/Softmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџЛ
window_attention/matmul_1BatchMatMulV2*window_attention/dropout/Identity:output:0)window_attention/strided_slice_2:output:0*
T0*/
_output_shapes
:џџџџџџџџџz
!window_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Г
window_attention/transpose_3	Transpose"window_attention/matmul_1:output:0*window_attention/transpose_3/perm:output:0*
T0*/
_output_shapes
:џџџџџџџџџu
 window_attention/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   Ј
window_attention/Reshape_3Reshape window_attention/transpose_3:y:0)window_attention/Reshape_3/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ќ
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
value	B : 
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
value	B : Ѓ
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
valueB: Й
'window_attention/dense_2/Tensordot/ProdProd4window_attention/dense_2/Tensordot/GatherV2:output:01window_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*window_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
)window_attention/dense_2/Tensordot/Prod_1Prod6window_attention/dense_2/Tensordot/GatherV2_1:output:03window_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.window_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)window_attention/dense_2/Tensordot/concatConcatV20window_attention/dense_2/Tensordot/free:output:00window_attention/dense_2/Tensordot/axes:output:07window_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
(window_attention/dense_2/Tensordot/stackPack0window_attention/dense_2/Tensordot/Prod:output:02window_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ш
,window_attention/dense_2/Tensordot/transpose	Transpose#window_attention/Reshape_3:output:02window_attention/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@е
*window_attention/dense_2/Tensordot/ReshapeReshape0window_attention/dense_2/Tensordot/transpose:y:01window_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџе
)window_attention/dense_2/Tensordot/MatMulMatMul3window_attention/dense_2/Tensordot/Reshape:output:09window_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@t
*window_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@r
0window_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+window_attention/dense_2/Tensordot/concat_1ConcatV24window_attention/dense_2/Tensordot/GatherV2:output:03window_attention/dense_2/Tensordot/Const_2:output:09window_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ю
"window_attention/dense_2/TensordotReshape3window_attention/dense_2/Tensordot/MatMul:product:04window_attention/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Є
/window_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOp8window_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ч
 window_attention/dense_2/BiasAddBiasAdd+window_attention/dense_2/Tensordot:output:07window_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@
#window_attention/dropout/Identity_1Identity)window_attention/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@h
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ      @   
	Reshape_4Reshape,window_attention/dropout/Identity_1:output:0Reshape_4/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@p
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*-
value$B""џџџџ            @   
	Reshape_5ReshapeReshape_4:output:0Reshape_5/shape:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@q
transpose_1/permConst*
_output_shapes
:*
dtype0*-
value$B""                   
transpose_1	TransposeReshape_5:output:0transpose_1/perm:output:0*
T0*7
_output_shapes%
#:!џџџџџџџџџ@h
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ        @   y
	Reshape_6Reshapetranspose_1:y:0Reshape_6/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @d
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ   @   y
	Reshape_7ReshapeReshape_6:output:0Reshape_7/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Q
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
valueB:
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
value	B :Ь
drop_path/random_uniform/shapePack drop_path/strided_slice:output:0)drop_path/random_uniform/shape/1:output:0)drop_path/random_uniform/shape/2:output:0*
N*
T0*
_output_shapes
:Ѓ
&drop_path/random_uniform/RandomUniformRandomUniform'drop_path/random_uniform/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0T
drop_path/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/addAddV2drop_path/add/x:output:0/drop_path/random_uniform/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџa
drop_path/FloorFloordrop_path/add:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
drop_path/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/truedivRealDivReshape_7:output:0drop_path/truediv/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@w
drop_path/mulMuldrop_path/truediv:z:0drop_path/Floor:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@Y
addAddV2xdrop_path/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:К
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:џџџџџџџџџЙ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ю
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'7Ф
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџЊ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Й
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ё
+sequential/dense_3/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: Ї
!sequential/dense_3/Tensordot/ProdProd.sequential/dense_3/Tensordot/GatherV2:output:0+sequential/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_3/Tensordot/Prod_1Prod0sequential/dense_3/Tensordot/GatherV2_1:output:0-sequential/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
#sequential/dense_3/Tensordot/concatConcatV2*sequential/dense_3/Tensordot/free:output:0*sequential/dense_3/Tensordot/axes:output:01sequential/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:В
"sequential/dense_3/Tensordot/stackPack*sequential/dense_3/Tensordot/Prod:output:0,sequential/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:У
&sequential/dense_3/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0,sequential/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@У
$sequential/dense_3/Tensordot/ReshapeReshape*sequential/dense_3/Tensordot/transpose:y:0+sequential/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџФ
#sequential/dense_3/Tensordot/MatMulMatMul-sequential/dense_3/Tensordot/Reshape:output:03sequential/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџo
$sequential/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*sequential/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
%sequential/dense_3/Tensordot/concat_1ConcatV2.sequential/dense_3/Tensordot/GatherV2:output:0-sequential/dense_3/Tensordot/Const_2:output:03sequential/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:О
sequential/dense_3/TensordotReshape-sequential/dense_3/Tensordot/MatMul:product:0.sequential/dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0З
sequential/dense_3/BiasAddBiasAdd%sequential/dense_3/Tensordot:output:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџe
 sequential/activation/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?­
sequential/activation/Gelu/mulMul)sequential/activation/Gelu/mul/x:output:0#sequential/dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџf
!sequential/activation/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?Ж
"sequential/activation/Gelu/truedivRealDiv#sequential/dense_3/BiasAdd:output:0*sequential/activation/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
sequential/activation/Gelu/ErfErf&sequential/activation/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџe
 sequential/activation/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ў
sequential/activation/Gelu/addAddV2)sequential/activation/Gelu/add/x:output:0"sequential/activation/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџЇ
 sequential/activation/Gelu/mul_1Mul"sequential/activation/Gelu/mul:z:0"sequential/activation/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџ
sequential/dropout_1/IdentityIdentity$sequential/activation/Gelu/mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџЁ
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : 
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
value	B : 
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
valueB: Ї
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:В
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:С
&sequential/dense_4/Tensordot/transpose	Transpose&sequential/dropout_1/Identity:output:0,sequential/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџУ
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџУ
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Н
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@
sequential/dropout_2/IdentityIdentity#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@g
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
valueB:
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
value	B :д
 drop_path/random_uniform_1/shapePack"drop_path/strided_slice_1:output:0+drop_path/random_uniform_1/shape/1:output:0+drop_path/random_uniform_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ї
(drop_path/random_uniform_1/RandomUniformRandomUniform)drop_path/random_uniform_1/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
dtype0V
drop_path/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/add_1AddV2drop_path/add_1/x:output:01drop_path/random_uniform_1/RandomUniform:output:0*
T0*+
_output_shapes
:џџџџџџџџџe
drop_path/Floor_1Floordrop_path/add_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
drop_path/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ьQx?
drop_path/truediv_1RealDiv&sequential/dropout_2/Identity:output:0drop_path/truediv_1/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@}
drop_path/mul_1Muldrop_path/truediv_1:z:0drop_path/Floor_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@c
add_1AddV2add:z:0drop_path/mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ@д
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp,^sequential/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp^window_attention/Gather*^window_attention/Reshape_1/ReadVariableOp0^window_attention/dense_1/BiasAdd/ReadVariableOp2^window_attention/dense_1/Tensordot/ReadVariableOp0^window_attention/dense_2/BiasAdd/ReadVariableOp2^window_attention/dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:џџџџџџџџџ@: : : : : : : : : : : : : : 2\
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
:џџџџџџџџџ@

_user_specified_namex

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
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
д
њ
B__inference_dense_8_layer_call_and_return_conditional_losses_12500

inputs4
!tensordot_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џV
й
G__inference_sequential_1_layer_call_and_return_conditional_losses_13072

inputs<
)dense_7_tensordot_readvariableop_resource:	@6
'dense_7_biasadd_readvariableop_resource:	<
)dense_8_tensordot_readvariableop_resource:	@5
'dense_8_biasadd_readvariableop_resource:@
identityЂdense_7/BiasAdd/ReadVariableOpЂ dense_7/Tensordot/ReadVariableOpЂdense_8/BiasAdd/ReadVariableOpЂ dense_8/Tensordot/ReadVariableOp
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : л
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
value	B : п
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
valueB: 
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_7/Tensordot/transpose	Transposeinputs!dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@Ђ
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЃ
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџd
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:џџџџџџџџџ\
activation_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
activation_1/Gelu/mulMul activation_1/Gelu/mul/x:output:0dense_7/BiasAdd:output:0*
T0*-
_output_shapes
:џџџџџџџџџ]
activation_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *ѓЕ?
activation_1/Gelu/truedivRealDivdense_7/BiasAdd:output:0!activation_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:џџџџџџџџџs
activation_1/Gelu/ErfErfactivation_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:џџџџџџџџџ\
activation_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
activation_1/Gelu/addAddV2 activation_1/Gelu/add/x:output:0activation_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:џџџџџџџџџ
activation_1/Gelu/mul_1Mulactivation_1/Gelu/mul:z:0activation_1/Gelu/add:z:0*
T0*-
_output_shapes
:џџџџџџџџџ\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?
dropout_4/dropout/MulMulactivation_1/Gelu/mul_1:z:0 dropout_4/dropout/Const:output:0*
T0*-
_output_shapes
:џџџџџџџџџb
dropout_4/dropout/ShapeShapeactivation_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:І
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Ъ
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:џџџџџџџџџ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:џџџџџџџџџ
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*-
_output_shapes
:џџџџџџџџџ
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes
:	@*
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
value	B : л
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
value	B : п
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
valueB: 
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
dense_8/Tensordot/transpose	Transposedropout_4/dropout/Mul_1:z:0!dense_8/Tensordot/concat:output:0*
T0*-
_output_shapes
:џџџџџџџџџЂ
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЂ
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ@\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?
dropout_5/dropout/MulMuldense_8/BiasAdd:output:0 dropout_5/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@_
dropout_5/dropout/ShapeShapedense_8/BiasAdd:output:0*
T0*
_output_shapes
:Ѕ
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Щ
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ@
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ@
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ@o
IdentityIdentitydropout_5/dropout/Mul_1:z:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@Ю
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Й&
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
§џџџџџџџџm
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџd
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
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
ўџџџџџџџџo
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџf
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ђ
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
 *  B`
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
 *  Bf
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
value	B : 

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
џџџџџџџџџY
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:Ђ

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*8
_output_shapes&
$:"џџџџџџџџџ@@џџџџџџџџџa
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"@   @   А
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
half_pixel_centers(
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*/
_output_shapes
:џџџџџџџџџ@@"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: :5 1
/
_output_shapes
:џџџџџџџџџ@@
Ж
{
+__inference_random_crop_layer_call_fn_11781

inputs
unknown:	
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_crop_layer_call_and_return_conditional_losses_10491w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@@: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ъ
ы
%__inference_model_layer_call_fn_10884
input_1
unknown:	
	unknown_0:	
	unknown_1:@
	unknown_2:@
	unknown_3:	@
	unknown_4:@
	unknown_5:@
	unknown_6:	@Р
	unknown_7:	Р
	unknown_8:	
	unknown_9:	

unknown_10:@@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:	@

unknown_15:	

unknown_16:	@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:	@Р

unknown_21:	Р

unknown_22:	

unknown_23:	!

unknown_24:

unknown_25:@@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:	@

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:


unknown_34:	W

unknown_35:W
identityЂStatefulPartitionedCallЕ
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
:џџџџџџџџџW*E
_read_only_resource_inputs'
%#	
 !"#$%*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10728o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџW`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:џџџџџџџџџ@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@@
!
_user_specified_name	input_1
Ђ

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_13155

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *qѕ?j
dropout/MulMulinputsdropout/Const:output:0*
T0*-
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Тѕ<Ќ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:џџџџџџџџџu
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:џџџџџџџџџo
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*-
_output_shapes
:џџџџџџџџџ_
IdentityIdentitydropout/Mul_1:z:0*
T0*-
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
Ї
E__inference_sequential_layer_call_and_return_conditional_losses_12233

inputs 
dense_3_12163:	@
dense_3_12165:	 
dense_4_12220:	@
dense_4_12222:@
identityЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallя
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_12163dense_3_12165*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_12162у
activation/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_12180м
dropout_1/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_12187
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_4_12220dense_4_12222*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_12219р
dropout_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_12230v
IdentityIdentity"dropout_2/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ@
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ@: : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

b
)__inference_dropout_4_layer_call_fn_13287

inputs
identityЂStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_12578u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж

'__inference_dense_7_layer_call_fn_13230

inputs
unknown:	@
	unknown_0:	
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_12443u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs"ПL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Г
serving_default
C
input_18
serving_default_input_1:0џџџџџџџџџ@@<
dense_100
StatefulPartitionedCall:0џџџџџџџџџWtensorflow/serving/predict:к
§
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
с
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
#_self_saveable_object_factories"
_tf_keras_layer
с
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_random_generator
#$_self_saveable_object_factories"
_tf_keras_layer
Ъ
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
#+_self_saveable_object_factories"
_tf_keras_layer
у
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

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

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
м
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Tlinear_trans
#U_self_saveable_object_factories"
_tf_keras_layer
Ъ
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
#\_self_saveable_object_factories"
_tf_keras_layer
р
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
Д
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
26
27
28
29
L30
31
32
c33
d34"
trackable_list_wrapper

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
25
26
27
28
29
c30
d31"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
в
trace_0
trace_1
trace_2
trace_32п
%__inference_model_layer_call_fn_10178
%__inference_model_layer_call_fn_11206
%__inference_model_layer_call_fn_11285
%__inference_model_layer_call_fn_10884Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
О
trace_0
trace_1
trace_2
trace_32Ы
@__inference_model_layer_call_and_return_conditional_losses_11411
@__inference_model_layer_call_and_return_conditional_losses_11769
@__inference_model_layer_call_and_return_conditional_losses_10967
@__inference_model_layer_call_and_return_conditional_losses_11054Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ЪBЧ
__inference__wrapped_model_9941input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
-
serving_default"
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
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ь
trace_0
trace_12
+__inference_random_crop_layer_call_fn_11774
+__inference_random_crop_layer_call_fn_11781Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12Ч
F__inference_random_crop_layer_call_and_return_conditional_losses_11827
F__inference_random_crop_layer_call_and_return_conditional_losses_12000Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1
/

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
В
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Ь
Ѓtrace_0
Єtrace_12
+__inference_random_flip_layer_call_fn_12005
+__inference_random_flip_layer_call_fn_12012Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zЃtrace_0zЄtrace_1

Ѕtrace_0
Іtrace_12Ч
F__inference_random_flip_layer_call_and_return_conditional_losses_12016
F__inference_random_flip_layer_call_and_return_conditional_losses_12125Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zЅtrace_0zІtrace_1
/
Ї
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
В
Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ш
­trace_02Щ
,__inference_patch_extract_layer_call_fn_3752
В
FullArgSpec
args

jimages
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z­trace_0

Ўtrace_02ф
G__inference_patch_extract_layer_call_and_return_conditional_losses_1291
В
FullArgSpec
args

jimages
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЎtrace_0
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
В
Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
щ
Дtrace_02Ъ
.__inference_patch_embedding_layer_call_fn_1272
В
FullArgSpec
args	
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zДtrace_0

Еtrace_02х
I__inference_patch_embedding_layer_call_and_return_conditional_losses_2689
В
FullArgSpec
args	
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0
ч
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses

fkernel
gbias
$М_self_saveable_object_factories"
_tf_keras_layer
с
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses
h
embeddings
$У_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper

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
В
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
Ф
Щtrace_0
Ъtrace_12
/__inference_swin_transformer_layer_call_fn_6442
/__inference_swin_transformer_layer_call_fn_2650Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЩtrace_0zЪtrace_1
љ
Ыtrace_0
Ьtrace_12О
I__inference_swin_transformer_layer_call_and_return_conditional_losses_721
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2087Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЫtrace_0zЬtrace_1
ё
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses
	гaxis
	igamma
jbeta
$д_self_saveable_object_factories"
_tf_keras_layer
П
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses
лqkv
мdropout
	нproj

kweight
 krelative_position_bias_table
vrelative_position_index
$о_self_saveable_object_factories"
_tf_keras_layer
б
п	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses
$х_self_saveable_object_factories"
_tf_keras_layer
ё
ц	variables
чtrainable_variables
шregularization_losses
щ	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses
	ьaxis
	pgamma
qbeta
$э_self_saveable_object_factories"
_tf_keras_layer
в
юlayer_with_weights-0
юlayer-0
яlayer-1
№layer-2
ёlayer_with_weights-1
ёlayer-3
ђlayer-4
ѓ	variables
єtrainable_variables
ѕregularization_losses
і	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses
$љ_self_saveable_object_factories"
_tf_keras_sequential
 "
trackable_dict_wrapper

w0
x1
y2
z3
{4
|5
}6
~7
8
9
10
11
12
L13
14"
trackable_list_wrapper

w0
x1
y2
z3
{4
|5
}6
~7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
В
њnon_trainable_variables
ћlayers
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
Ш
џtrace_0
trace_12
1__inference_swin_transformer_1_layer_call_fn_1176
1__inference_swin_transformer_1_layer_call_fn_1623Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zџtrace_0ztrace_1
ў
trace_0
trace_12У
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_2354
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_6078Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ё
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	wgamma
xbeta
$_self_saveable_object_factories"
_tf_keras_layer
Р
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
qkv
dropout
	proj

yweight
 yrelative_position_bias_table
relative_position_index
$_self_saveable_object_factories"
_tf_keras_layer
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$_self_saveable_object_factories"
_tf_keras_layer
ё
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses
	Ђaxis
	~gamma
beta
$Ѓ_self_saveable_object_factories"
_tf_keras_layer
в
Єlayer_with_weights-0
Єlayer-0
Ѕlayer-1
Іlayer-2
Їlayer_with_weights-1
Їlayer-3
Јlayer-4
Љ	variables
Њtrainable_variables
Ћregularization_losses
Ќ	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses
$Џ_self_saveable_object_factories"
_tf_keras_sequential
0:.2swin_transformer_1/Variable
 "
trackable_dict_wrapper
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
у
Еtrace_02Ф
,__inference_patch_merging_layer_call_fn_2744
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0
ў
Жtrace_02п
G__inference_patch_merging_layer_call_and_return_conditional_losses_2938
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЖtrace_0
о
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses
kernel
$Н_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object

Уtrace_02ь
8__inference_global_average_pooling1d_layer_call_fn_12692Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zУtrace_0
І
Фtrace_02
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_12698Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zФtrace_0
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
В
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
ю
Ъtrace_02Я
(__inference_dense_10_layer_call_fn_12707Ђ
В
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
annotationsЊ *
 zЪtrace_0

Ыtrace_02ъ
C__inference_dense_10_layer_call_and_return_conditional_losses_12718Ђ
В
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
annotationsЊ *
 zЫtrace_0
": 	W2dense_10/kernel
:W2dense_10/bias
 "
trackable_dict_wrapper
.:,@2patch_embedding/dense/kernel
(:&@2patch_embedding/dense/bias
7:5	@2$patch_embedding/embedding/embeddings
8:6@2*swin_transformer/layer_normalization/gamma
7:5@2)swin_transformer/layer_normalization/beta
::8	2(swin_transformer/window_attention/weight
C:A	@Р20swin_transformer/window_attention/dense_1/kernel
=:;Р2.swin_transformer/window_attention/dense_1/bias
B:@@@20swin_transformer/window_attention/dense_2/kernel
<::@2.swin_transformer/window_attention/dense_2/bias
::8@2,swin_transformer/layer_normalization_1/gamma
9:7@2+swin_transformer/layer_normalization_1/beta
!:	@2dense_3/kernel
:2dense_3/bias
!:	@2dense_4/kernel
:@2dense_4/bias
::8	2*swin_transformer/window_attention/Variable
<::@2.swin_transformer_1/layer_normalization_2/gamma
;:9@2-swin_transformer_1/layer_normalization_2/beta
>:<	2,swin_transformer_1/window_attention_1/weight
G:E	@Р24swin_transformer_1/window_attention_1/dense_5/kernel
A:?Р22swin_transformer_1/window_attention_1/dense_5/bias
F:D@@24swin_transformer_1/window_attention_1/dense_6/kernel
@:>@22swin_transformer_1/window_attention_1/dense_6/bias
<::@2.swin_transformer_1/layer_normalization_3/gamma
;:9@2-swin_transformer_1/layer_normalization_3/beta
!:	@2dense_7/kernel
:2dense_7/bias
!:	@2dense_8/kernel
:@2dense_8/bias
>:<	2.swin_transformer_1/window_attention_1/Variable
0:.
2patch_merging/dense_9/kernel
6
v0
L1
2"
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
Ь0
Э1
Ю2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
јBѕ
%__inference_model_layer_call_fn_10178input_1"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
їBє
%__inference_model_layer_call_fn_11206inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
їBє
%__inference_model_layer_call_fn_11285inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
јBѕ
%__inference_model_layer_call_fn_10884input_1"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_11411inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_11769inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_10967input_1"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_11054input_1"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ЪBЧ
#__inference_signature_wrapper_11131input_1"
В
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
annotationsЊ *
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
ёBю
+__inference_random_crop_layer_call_fn_11774inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ёBю
+__inference_random_crop_layer_call_fn_11781inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
F__inference_random_crop_layer_call_and_return_conditional_losses_11827inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
F__inference_random_crop_layer_call_and_return_conditional_losses_12000inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
/
Я
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
ёBю
+__inference_random_flip_layer_call_fn_12005inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ёBю
+__inference_random_flip_layer_call_fn_12012inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
F__inference_random_flip_layer_call_and_return_conditional_losses_12016inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
F__inference_random_flip_layer_call_and_return_conditional_losses_12125inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
/
а
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
ЬBЩ
,__inference_patch_extract_layer_call_fn_3752"
В
FullArgSpec
args

jimages
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
чBф
G__inference_patch_extract_layer_call_and_return_conditional_losses_1291"
В
FullArgSpec
args

jimages
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
ЭBЪ
.__inference_patch_embedding_layer_call_fn_1272"
В
FullArgSpec
args	
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
шBх
I__inference_patch_embedding_layer_call_and_return_conditional_losses_2689"
В
FullArgSpec
args	
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
И
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
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
И
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
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
лBи
/__inference_swin_transformer_layer_call_fn_6442"Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
лBи
/__inference_swin_transformer_layer_call_fn_2650"Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕBђ
I__inference_swin_transformer_layer_call_and_return_conditional_losses_721"Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2087"Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
И
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
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
И
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
Ж2ГА
ЉВЅ
FullArgSpec$
args
jx
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ж2ГА
ЉВЅ
FullArgSpec$
args
jx
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ч
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses

lkernel
mbias
$ы_self_saveable_object_factories"
_tf_keras_layer
щ
ь	variables
эtrainable_variables
юregularization_losses
я	keras_api
№__call__
+ё&call_and_return_all_conditional_losses
ђ_random_generator
$ѓ_self_saveable_object_factories"
_tf_keras_layer
ч
є	variables
ѕtrainable_variables
іregularization_losses
ї	keras_api
ј__call__
+љ&call_and_return_all_conditional_losses

nkernel
obias
$њ_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
п	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ц	variables
чtrainable_variables
шregularization_losses
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ч
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

rkernel
sbias
$_self_saveable_object_factories"
_tf_keras_layer
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
$_self_saveable_object_factories"
_tf_keras_layer
щ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator
$_self_saveable_object_factories"
_tf_keras_layer
ч
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses

tkernel
ubias
$Ё_self_saveable_object_factories"
_tf_keras_layer
щ
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
Ѕ	keras_api
І__call__
+Ї&call_and_return_all_conditional_losses
Ј_random_generator
$Љ_self_saveable_object_factories"
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
И
Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
ѓ	variables
єtrainable_variables
ѕregularization_losses
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
ц
Џtrace_0
Аtrace_1
Бtrace_2
Вtrace_32ѓ
*__inference_sequential_layer_call_fn_12244
*__inference_sequential_layer_call_fn_12741
*__inference_sequential_layer_call_fn_12754
*__inference_sequential_layer_call_fn_12372Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 zЏtrace_0zАtrace_1zБtrace_2zВtrace_3
в
Гtrace_0
Дtrace_1
Еtrace_2
Жtrace_32п
E__inference_sequential_layer_call_and_return_conditional_losses_12820
E__inference_sequential_layer_call_and_return_conditional_losses_12900
E__inference_sequential_layer_call_and_return_conditional_losses_12389
E__inference_sequential_layer_call_and_return_conditional_losses_12406Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 zГtrace_0zДtrace_1zЕtrace_2zЖtrace_3
 "
trackable_dict_wrapper
/
L0
1"
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
нBк
1__inference_swin_transformer_1_layer_call_fn_1176"Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
нBк
1__inference_swin_transformer_1_layer_call_fn_1623"Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_2354"Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_6078"Є
В
FullArgSpec
args
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
И
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
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
5"
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
И
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ж2ГА
ЉВЅ
FullArgSpec$
args
jx
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ж2ГА
ЉВЅ
FullArgSpec$
args
jx
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ч
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses

zkernel
{bias
$Ч_self_saveable_object_factories"
_tf_keras_layer
щ
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses
Ю_random_generator
$Я_self_saveable_object_factories"
_tf_keras_layer
ч
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses

|kernel
}bias
$ж_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
И
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щ
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses
kernel
	bias
$ч_self_saveable_object_factories"
_tf_keras_layer
б
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses
$ю_self_saveable_object_factories"
_tf_keras_layer
щ
я	variables
№trainable_variables
ёregularization_losses
ђ	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses
ѕ_random_generator
$і_self_saveable_object_factories"
_tf_keras_layer
щ
ї	variables
јtrainable_variables
љregularization_losses
њ	keras_api
ћ__call__
+ќ&call_and_return_all_conditional_losses
kernel
	bias
$§_self_saveable_object_factories"
_tf_keras_layer
щ
ў	variables
џtrainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator
$_self_saveable_object_factories"
_tf_keras_layer
@
0
1
2
3"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Љ	variables
Њtrainable_variables
Ћregularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
ю
trace_0
trace_1
trace_2
trace_32ћ
,__inference_sequential_1_layer_call_fn_12525
,__inference_sequential_1_layer_call_fn_12913
,__inference_sequential_1_layer_call_fn_12926
,__inference_sequential_1_layer_call_fn_12653Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
к
trace_0
trace_1
trace_2
trace_32ч
G__inference_sequential_1_layer_call_and_return_conditional_losses_12992
G__inference_sequential_1_layer_call_and_return_conditional_losses_13072
G__inference_sequential_1_layer_call_and_return_conditional_losses_12670
G__inference_sequential_1_layer_call_and_return_conditional_losses_12687Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
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
ЧBФ
,__inference_patch_merging_layer_call_fn_2744"
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
тBп
G__inference_patch_merging_layer_call_and_return_conditional_losses_2938"
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
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
љBі
8__inference_global_average_pooling1d_layer_call_fn_12692inputs"Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_12698inputs"Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
мBй
(__inference_dense_10_layer_call_fn_12707inputs"Ђ
В
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
annotationsЊ *
 
їBє
C__inference_dense_10_layer_call_and_return_conditional_losses_12718inputs"Ђ
В
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
annotationsЊ *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count
 
_fn_kwargs"
_tf_keras_metric
c
Ё	variables
Ђ	keras_api

Ѓtotal

Єcount
Ѕ
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
л0
м1
н2"
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
И
Іnon_trainable_variables
Їlayers
Јmetrics
 Љlayer_regularization_losses
Њlayer_metrics
х	variables
цtrainable_variables
чregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
ь	variables
эtrainable_variables
юregularization_losses
№__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
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
И
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
є	variables
ѕtrainable_variables
іregularization_losses
ј__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
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
И
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
э
Кtrace_02Ю
'__inference_dense_3_layer_call_fn_13081Ђ
В
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
annotationsЊ *
 zКtrace_0

Лtrace_02щ
B__inference_dense_3_layer_call_and_return_conditional_losses_13111Ђ
В
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
annotationsЊ *
 zЛtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
№
Сtrace_02б
*__inference_activation_layer_call_fn_13116Ђ
В
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
annotationsЊ *
 zСtrace_0

Тtrace_02ь
E__inference_activation_layer_call_and_return_conditional_losses_13128Ђ
В
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
annotationsЊ *
 zТtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ш
Шtrace_0
Щtrace_12
)__inference_dropout_1_layer_call_fn_13133
)__inference_dropout_1_layer_call_fn_13138Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zШtrace_0zЩtrace_1
ў
Ъtrace_0
Ыtrace_12У
D__inference_dropout_1_layer_call_and_return_conditional_losses_13143
D__inference_dropout_1_layer_call_and_return_conditional_losses_13155Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zЪtrace_0zЫtrace_1
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
И
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
э
бtrace_02Ю
'__inference_dense_4_layer_call_fn_13164Ђ
В
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
annotationsЊ *
 zбtrace_0

вtrace_02щ
B__inference_dense_4_layer_call_and_return_conditional_losses_13194Ђ
В
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
annotationsЊ *
 zвtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
Ђ	variables
Ѓtrainable_variables
Єregularization_losses
І__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
Ш
иtrace_0
йtrace_12
)__inference_dropout_2_layer_call_fn_13199
)__inference_dropout_2_layer_call_fn_13204Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zиtrace_0zйtrace_1
ў
кtrace_0
лtrace_12У
D__inference_dropout_2_layer_call_and_return_conditional_losses_13209
D__inference_dropout_2_layer_call_and_return_conditional_losses_13221Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zкtrace_0zлtrace_1
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
H
ю0
я1
№2
ё3
ђ4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
*__inference_sequential_layer_call_fn_12244dense_3_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ќBљ
*__inference_sequential_layer_call_fn_12741inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ќBљ
*__inference_sequential_layer_call_fn_12754inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
*__inference_sequential_layer_call_fn_12372dense_3_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_12820inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_12900inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_12389dense_3_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_12406dense_3_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
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
0"
trackable_list_wrapper
8
0
1
2"
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
И
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
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
И
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
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
annotationsЊ *
 
Ј2ЅЂ
В
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
annotationsЊ *
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
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
э
№trace_02Ю
'__inference_dense_7_layer_call_fn_13230Ђ
В
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
annotationsЊ *
 z№trace_0

ёtrace_02щ
B__inference_dense_7_layer_call_and_return_conditional_losses_13260Ђ
В
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
annotationsЊ *
 zёtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
ђ
їtrace_02г
,__inference_activation_1_layer_call_fn_13265Ђ
В
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
annotationsЊ *
 zїtrace_0

јtrace_02ю
G__inference_activation_1_layer_call_and_return_conditional_losses_13277Ђ
В
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
annotationsЊ *
 zјtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
я	variables
№trainable_variables
ёregularization_losses
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
Ш
ўtrace_0
џtrace_12
)__inference_dropout_4_layer_call_fn_13282
)__inference_dropout_4_layer_call_fn_13287Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zўtrace_0zџtrace_1
ў
trace_0
trace_12У
D__inference_dropout_4_layer_call_and_return_conditional_losses_13292
D__inference_dropout_4_layer_call_and_return_conditional_losses_13304Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ї	variables
јtrainable_variables
љregularization_losses
ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_dense_8_layer_call_fn_13313Ђ
В
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
annotationsЊ *
 ztrace_0

trace_02щ
B__inference_dense_8_layer_call_and_return_conditional_losses_13343Ђ
В
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
annotationsЊ *
 ztrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ў	variables
џtrainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ш
trace_0
trace_12
)__inference_dropout_5_layer_call_fn_13348
)__inference_dropout_5_layer_call_fn_13353Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1
ў
trace_0
trace_12У
D__inference_dropout_5_layer_call_and_return_conditional_losses_13358
D__inference_dropout_5_layer_call_and_return_conditional_losses_13370Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
H
Є0
Ѕ1
І2
Ї3
Ј4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_1_layer_call_fn_12525dense_7_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ўBћ
,__inference_sequential_1_layer_call_fn_12913inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ўBћ
,__inference_sequential_1_layer_call_fn_12926inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
,__inference_sequential_1_layer_call_fn_12653dense_7_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
G__inference_sequential_1_layer_call_and_return_conditional_losses_12992inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
B
G__inference_sequential_1_layer_call_and_return_conditional_losses_13072inputs"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
 B
G__inference_sequential_1_layer_call_and_return_conditional_losses_12670dense_7_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
 B
G__inference_sequential_1_layer_call_and_return_conditional_losses_12687dense_7_input"Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
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
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ѓ0
Є1"
trackable_list_wrapper
.
Ё	variables"
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
лBи
'__inference_dense_3_layer_call_fn_13081inputs"Ђ
В
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
annotationsЊ *
 
іBѓ
B__inference_dense_3_layer_call_and_return_conditional_losses_13111inputs"Ђ
В
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
annotationsЊ *
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
оBл
*__inference_activation_layer_call_fn_13116inputs"Ђ
В
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
annotationsЊ *
 
љBі
E__inference_activation_layer_call_and_return_conditional_losses_13128inputs"Ђ
В
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
annotationsЊ *
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
яBь
)__inference_dropout_1_layer_call_fn_13133inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
яBь
)__inference_dropout_1_layer_call_fn_13138inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_13143inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_13155inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
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
лBи
'__inference_dense_4_layer_call_fn_13164inputs"Ђ
В
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
annotationsЊ *
 
іBѓ
B__inference_dense_4_layer_call_and_return_conditional_losses_13194inputs"Ђ
В
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
annotationsЊ *
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
яBь
)__inference_dropout_2_layer_call_fn_13199inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
яBь
)__inference_dropout_2_layer_call_fn_13204inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
D__inference_dropout_2_layer_call_and_return_conditional_losses_13209inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
D__inference_dropout_2_layer_call_and_return_conditional_losses_13221inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
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
лBи
'__inference_dense_7_layer_call_fn_13230inputs"Ђ
В
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
annotationsЊ *
 
іBѓ
B__inference_dense_7_layer_call_and_return_conditional_losses_13260inputs"Ђ
В
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
annotationsЊ *
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
рBн
,__inference_activation_1_layer_call_fn_13265inputs"Ђ
В
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
annotationsЊ *
 
ћBј
G__inference_activation_1_layer_call_and_return_conditional_losses_13277inputs"Ђ
В
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
annotationsЊ *
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
яBь
)__inference_dropout_4_layer_call_fn_13282inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
яBь
)__inference_dropout_4_layer_call_fn_13287inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
D__inference_dropout_4_layer_call_and_return_conditional_losses_13292inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
D__inference_dropout_4_layer_call_and_return_conditional_losses_13304inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
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
лBи
'__inference_dense_8_layer_call_fn_13313inputs"Ђ
В
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
annotationsЊ *
 
іBѓ
B__inference_dense_8_layer_call_and_return_conditional_losses_13343inputs"Ђ
В
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
annotationsЊ *
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
яBь
)__inference_dropout_5_layer_call_fn_13348inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
яBь
)__inference_dropout_5_layer_call_fn_13353inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
D__inference_dropout_5_layer_call_and_return_conditional_losses_13358inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
D__inference_dropout_5_layer_call_and_return_conditional_losses_13370inputs"Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 О
__inference__wrapped_model_9941)fghijlmvknopqrstuwxz{yL|}~cd8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ@@
Њ "3Њ0
.
dense_10"
dense_10џџџџџџџџџWЏ
G__inference_activation_1_layer_call_and_return_conditional_losses_13277d5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "+Ђ(
!
0џџџџџџџџџ
 
,__inference_activation_1_layer_call_fn_13265W5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "џџџџџџџџџ­
E__inference_activation_layer_call_and_return_conditional_losses_13128d5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "+Ђ(
!
0џџџџџџџџџ
 
*__inference_activation_layer_call_fn_13116W5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "џџџџџџџџџЄ
C__inference_dense_10_layer_call_and_return_conditional_losses_12718]cd0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџW
 |
(__inference_dense_10_layer_call_fn_12707Pcd0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџW­
B__inference_dense_3_layer_call_and_return_conditional_losses_13111grs4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "+Ђ(
!
0џџџџџџџџџ
 
'__inference_dense_3_layer_call_fn_13081Zrs4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ­
B__inference_dense_4_layer_call_and_return_conditional_losses_13194gtu5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "*Ђ'
 
0џџџџџџџџџ@
 
'__inference_dense_4_layer_call_fn_13164Ztu5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Џ
B__inference_dense_7_layer_call_and_return_conditional_losses_13260i4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "+Ђ(
!
0џџџџџџџџџ
 
'__inference_dense_7_layer_call_fn_13230\4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЏ
B__inference_dense_8_layer_call_and_return_conditional_losses_13343i5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "*Ђ'
 
0џџџџџџџџџ@
 
'__inference_dense_8_layer_call_fn_13313\5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@А
D__inference_dropout_1_layer_call_and_return_conditional_losses_13143h9Ђ6
/Ђ,
&#
inputsџџџџџџџџџ
p 
Њ "+Ђ(
!
0џџџџџџџџџ
 А
D__inference_dropout_1_layer_call_and_return_conditional_losses_13155h9Ђ6
/Ђ,
&#
inputsџџџџџџџџџ
p
Њ "+Ђ(
!
0џџџџџџџџџ
 
)__inference_dropout_1_layer_call_fn_13133[9Ђ6
/Ђ,
&#
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
)__inference_dropout_1_layer_call_fn_13138[9Ђ6
/Ђ,
&#
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЎ
D__inference_dropout_2_layer_call_and_return_conditional_losses_13209f8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ@
p 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ў
D__inference_dropout_2_layer_call_and_return_conditional_losses_13221f8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ@
p
Њ "*Ђ'
 
0џџџџџџџџџ@
 
)__inference_dropout_2_layer_call_fn_13199Y8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@
)__inference_dropout_2_layer_call_fn_13204Y8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ@
p
Њ "џџџџџџџџџ@А
D__inference_dropout_4_layer_call_and_return_conditional_losses_13292h9Ђ6
/Ђ,
&#
inputsџџџџџџџџџ
p 
Њ "+Ђ(
!
0џџџџџџџџџ
 А
D__inference_dropout_4_layer_call_and_return_conditional_losses_13304h9Ђ6
/Ђ,
&#
inputsџџџџџџџџџ
p
Њ "+Ђ(
!
0џџџџџџџџџ
 
)__inference_dropout_4_layer_call_fn_13282[9Ђ6
/Ђ,
&#
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
)__inference_dropout_4_layer_call_fn_13287[9Ђ6
/Ђ,
&#
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЎ
D__inference_dropout_5_layer_call_and_return_conditional_losses_13358f8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ@
p 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ў
D__inference_dropout_5_layer_call_and_return_conditional_losses_13370f8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ@
p
Њ "*Ђ'
 
0џџџџџџџџџ@
 
)__inference_dropout_5_layer_call_fn_13348Y8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@
)__inference_dropout_5_layer_call_fn_13353Y8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ@
p
Њ "џџџџџџџџџ@в
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_12698{IЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Њ
8__inference_global_average_pooling1d_layer_call_fn_12692nIЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ "!џџџџџџџџџџџџџџџџџџй
@__inference_model_layer_call_and_return_conditional_losses_10967)fghijlmvknopqrstuwxz{yL|}~cd@Ђ=
6Ђ3
)&
input_1џџџџџџџџџ@@
p 

 
Њ "%Ђ"

0џџџџџџџџџW
 н
@__inference_model_layer_call_and_return_conditional_losses_11054-Яаfghijlmvknopqrstuwxz{yL|}~cd@Ђ=
6Ђ3
)&
input_1џџџџџџџџџ@@
p

 
Њ "%Ђ"

0џџџџџџџџџW
 и
@__inference_model_layer_call_and_return_conditional_losses_11411)fghijlmvknopqrstuwxz{yL|}~cd?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@@
p 

 
Њ "%Ђ"

0џџџџџџџџџW
 м
@__inference_model_layer_call_and_return_conditional_losses_11769-Яаfghijlmvknopqrstuwxz{yL|}~cd?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@@
p

 
Њ "%Ђ"

0џџџџџџџџџW
 Б
%__inference_model_layer_call_fn_10178)fghijlmvknopqrstuwxz{yL|}~cd@Ђ=
6Ђ3
)&
input_1џџџџџџџџџ@@
p 

 
Њ "џџџџџџџџџWЕ
%__inference_model_layer_call_fn_10884-Яаfghijlmvknopqrstuwxz{yL|}~cd@Ђ=
6Ђ3
)&
input_1џџџџџџџџџ@@
p

 
Њ "џџџџџџџџџWА
%__inference_model_layer_call_fn_11206)fghijlmvknopqrstuwxz{yL|}~cd?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@@
p 

 
Њ "џџџџџџџџџWД
%__inference_model_layer_call_fn_11285-Яаfghijlmvknopqrstuwxz{yL|}~cd?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@@
p

 
Њ "џџџџџџџџџWГ
I__inference_patch_embedding_layer_call_and_return_conditional_losses_2689ffgh3Ђ0
)Ђ&
$!
patchџџџџџџџџџ
Њ "*Ђ'
 
0џџџџџџџџџ@
 
.__inference_patch_embedding_layer_call_fn_1272Yfgh3Ђ0
)Ђ&
$!
patchџџџџџџџџџ
Њ "џџџџџџџџџ@А
G__inference_patch_extract_layer_call_and_return_conditional_losses_1291e7Ђ4
-Ђ*
(%
imagesџџџџџџџџџ@@
Њ "*Ђ'
 
0џџџџџџџџџ
 
,__inference_patch_extract_layer_call_fn_3752X7Ђ4
-Ђ*
(%
imagesџџџџџџџџџ@@
Њ "џџџџџџџџџ­
G__inference_patch_merging_layer_call_and_return_conditional_losses_2938b/Ђ,
%Ђ"
 
xџџџџџџџџџ@
Њ "+Ђ(
!
0џџџџџџџџџ
 
,__inference_patch_merging_layer_call_fn_2744U/Ђ,
%Ђ"
 
xџџџџџџџџџ@
Њ "џџџџџџџџџЖ
F__inference_random_crop_layer_call_and_return_conditional_losses_11827l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 К
F__inference_random_crop_layer_call_and_return_conditional_losses_12000pЯ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 
+__inference_random_crop_layer_call_fn_11774_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ " џџџџџџџџџ@@
+__inference_random_crop_layer_call_fn_11781cЯ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ " џџџџџџџџџ@@Ж
F__inference_random_flip_layer_call_and_return_conditional_losses_12016l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 К
F__inference_random_flip_layer_call_and_return_conditional_losses_12125pа;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ "-Ђ*
# 
0џџџџџџџџџ@@
 
+__inference_random_flip_layer_call_fn_12005_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ " џџџџџџџџџ@@
+__inference_random_flip_layer_call_fn_12012cа;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ " џџџџџџџџџ@@Ц
G__inference_sequential_1_layer_call_and_return_conditional_losses_12670{CЂ@
9Ђ6
,)
dense_7_inputџџџџџџџџџ@
p 

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ц
G__inference_sequential_1_layer_call_and_return_conditional_losses_12687{CЂ@
9Ђ6
,)
dense_7_inputџџџџџџџџџ@
p

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 П
G__inference_sequential_1_layer_call_and_return_conditional_losses_12992t<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p 

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 П
G__inference_sequential_1_layer_call_and_return_conditional_losses_13072t<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 
,__inference_sequential_1_layer_call_fn_12525nCЂ@
9Ђ6
,)
dense_7_inputџџџџџџџџџ@
p 

 
Њ "џџџџџџџџџ@
,__inference_sequential_1_layer_call_fn_12653nCЂ@
9Ђ6
,)
dense_7_inputџџџџџџџџџ@
p

 
Њ "џџџџџџџџџ@
,__inference_sequential_1_layer_call_fn_12913g<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p 

 
Њ "џџџџџџџџџ@
,__inference_sequential_1_layer_call_fn_12926g<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p

 
Њ "џџџџџџџџџ@Р
E__inference_sequential_layer_call_and_return_conditional_losses_12389wrstuCЂ@
9Ђ6
,)
dense_3_inputџџџџџџџџџ@
p 

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Р
E__inference_sequential_layer_call_and_return_conditional_losses_12406wrstuCЂ@
9Ђ6
,)
dense_3_inputџџџџџџџџџ@
p

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Й
E__inference_sequential_layer_call_and_return_conditional_losses_12820prstu<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p 

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Й
E__inference_sequential_layer_call_and_return_conditional_losses_12900prstu<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p

 
Њ "*Ђ'
 
0џџџџџџџџџ@
 
*__inference_sequential_layer_call_fn_12244jrstuCЂ@
9Ђ6
,)
dense_3_inputџџџџџџџџџ@
p 

 
Њ "џџџџџџџџџ@
*__inference_sequential_layer_call_fn_12372jrstuCЂ@
9Ђ6
,)
dense_3_inputџџџџџџџџџ@
p

 
Њ "џџџџџџџџџ@
*__inference_sequential_layer_call_fn_12741crstu<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p 

 
Њ "џџџџџџџџџ@
*__inference_sequential_layer_call_fn_12754crstu<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ@
p

 
Њ "џџџџџџџџџ@Э
#__inference_signature_wrapper_11131Ѕ)fghijlmvknopqrstuwxz{yL|}~cdCЂ@
Ђ 
9Њ6
4
input_1)&
input_1џџџџџџџџџ@@"3Њ0
.
dense_10"
dense_10џџџџџџџџџWЧ
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_2354wwxz{yL|}~3Ђ0
)Ђ&
 
xџџџџџџџџџ@
p 
Њ "*Ђ'
 
0џџџџџџџџџ@
 Ч
L__inference_swin_transformer_1_layer_call_and_return_conditional_losses_6078wwxz{yL|}~3Ђ0
)Ђ&
 
xџџџџџџџџџ@
p
Њ "*Ђ'
 
0џџџџџџџџџ@
 
1__inference_swin_transformer_1_layer_call_fn_1176jwxz{yL|}~3Ђ0
)Ђ&
 
xџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@
1__inference_swin_transformer_1_layer_call_fn_1623jwxz{yL|}~3Ђ0
)Ђ&
 
xџџџџџџџџџ@
p
Њ "џџџџџџџџџ@П
J__inference_swin_transformer_layer_call_and_return_conditional_losses_2087qijlmvknopqrstu3Ђ0
)Ђ&
 
xџџџџџџџџџ@
p
Њ "*Ђ'
 
0џџџџџџџџџ@
 О
I__inference_swin_transformer_layer_call_and_return_conditional_losses_721qijlmvknopqrstu3Ђ0
)Ђ&
 
xџџџџџџџџџ@
p 
Њ "*Ђ'
 
0џџџџџџџџџ@
 
/__inference_swin_transformer_layer_call_fn_2650dijlmvknopqrstu3Ђ0
)Ђ&
 
xџџџџџџџџџ@
p
Њ "џџџџџџџџџ@
/__inference_swin_transformer_layer_call_fn_6442dijlmvknopqrstu3Ђ0
)Ђ&
 
xџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@