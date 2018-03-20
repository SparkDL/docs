# inference调用链分析

以最简单的lenet5为例, 探究inference过程的调用链

示例代码位于'spark/dl/src/main/scala/com/pzque/sparkdl/lenet', 模型的checkpoint已保存好, 下载好数据后可以直接运行'Test.scala'查看测试集上的推断准确率.

## lenet模型定义
首先看一下lenet模型的定义.

`apply`和`graph`函数分别使用了Sequential和Graph的API定义模型, 二者是等价的.

模型的结构非常简单, 在测试集上可以达到98.93%的准确率.

```scala
28*28 -> (Conv -> MaxPooling)*2 -> (FullConnected)*2 -> LogSoftMax
``` 

```scala
object LeNet5 {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5"))
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, classNum).setName("fc2"))
      .add(LogSoftMax())
  }
  def graph(classNum: Int): Module[Float] = {
    val input = Reshape(Array(1, 28, 28)).inputs()
    val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5").inputs(input)
    val tanh1 = Tanh().inputs(conv1)
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).inputs(tanh1)
    val tanh2 = Tanh().inputs(pool1)
    val conv2 = SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5").inputs(tanh2)
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).inputs(conv2)
    val reshape = Reshape(Array(12 * 4 * 4)).inputs(pool2)
    val fc1 = Linear(12 * 4 * 4, 100).setName("fc1").inputs(reshape)
    val tanh3 = Tanh().inputs(fc1)
    val fc2 = Linear(100, classNum).setName("fc2").inputs(tanh3)
    val output = LogSoftMax().inputs(fc2)

    Graph(input, output)
  }
}
```

## inference调用链
infrence的核心代码如下: 
```scala
      // 加载测试数据, 调用SparkContext类的parallize方法将其转为RDD
      val rddData: RDD[ByteRecord] = sc.parallelize(load(validationData, validationLabel), partitionNum)

      // 定义一个数据预处理器, 将ByteRecord格式转为Sample[Float]
      val transformer: Transformer[ByteRecord, Sample[Float]] =
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToSample()

      // 使用transformer构造验证集RDD
      val evaluationSet: RDD[Sample[Float]] = transformer(rddData)

      // 加载模型
      val model = Module.load[Float](param.model)

      // 执行模型, 获取结果
      val result = model.evaluate(evaluationSet,
        Array(new Top1Accuracy[Float]), Some(param.batchSize))
```

前面的一堆都是使用spark的RDD进行数据预处理与转换, 最后得到`evaluationSet`, 也是一个RDD, 元素是`Sample[Flaot]`的类型.

我们看到其先是通过`Module.load[Float]`将模型加载进来, 然后利用模型执行evaluate操作.

我们需要关注这一句:

```scala
model.evaluate(evaluationSet,
        Array(new Top1Accuracy[Float]), 
        Some(param.batchSize))
```

找到它的定义, 位于`AbstractModule`类:
```scala
  /**
   * use ValidationMethod to evaluate module on the given rdd dataset
   * @param dataset dataset for test
   * @param vMethods validation methods
   * @param batchSize total batchsize of all partitions,
   *                  optional param and default 4 * partitionNum of dataset
   * @return
   */
  final def evaluate(
    dataset: RDD[Sample[T]],
    vMethods: Array[ValidationMethod[T]],
    batchSize: Option[Int] = None
  ): Array[(ValidationResult, ValidationMethod[T])] = {
    Evaluator(this).test(dataset, vMethods, batchSize)
  }
```


三个参数,

- `dataset`: 是你要运行模型的数据集
- `vMethods`: 是最后模型运行完成运行的一些统计工作, 比如这里的Top1Accuracy就是统计一下准确率 
- `batchSize`: 注意这个不是机器学习的那个batchsize(每个batch的大小), 而是将全部的数据集分成多少batch

然后最后执行模型的代码就是`Evaluator(this).test(dataset, vMethods, batchSize)`了, 下面来看一下它的实现.

## Evaluator

 ```scala
/**
 * model evaluator
 * @param model model to be evaluated
 */
class Evaluator[T: ClassTag] private[optim](model: Module[T])(implicit ev: TensorNumeric[T])
  extends Serializable {

  private val batchPerPartition = 4

  /**
   * Applies ValidationMethod to the model and rdd dataset.
   * @param vMethods
   * @param batchSize total batchsize
   * @return
   */
  def test(dataset: RDD[Sample[T]],
   vMethods: Array[ValidationMethod[T]],
   batchSize: Option[Int] = None): Array[(ValidationResult, ValidationMethod[T])] = {

    val modelBroad = ModelBroadcast[T]().broadcast(dataset.sparkContext, model.evaluate())
    val partitionNum = dataset.partitions.length

    val totalBatch = batchSize.getOrElse(batchPerPartition * partitionNum)
    val otherBroad = dataset.sparkContext.broadcast(vMethods, SampleToMiniBatch(
      batchSize = totalBatch, partitionNum = Some(partitionNum)))

    dataset.mapPartitions(partition => {
      val localModel = modelBroad.value()
      val localMethod = otherBroad.value._1.map(_.clone())
      val localTransformer = otherBroad.value._2.cloneTransformer()
      val miniBatch = localTransformer(partition)
      miniBatch.map(batch => {
        val output = localModel.forward(batch.getInput())
        localMethod.map(validation => {
          validation(output, batch.getTarget())
        })
      })
    }).reduce((left, right) => {
        left.zip(right).map { case (l, r) => l + r }
    }).zip(vMethods)
  }
}
```

上面是这个类的全部代码, 这个类也只是在全局做调度, 很简单. 具体的执行逻辑当然还是在`AbstractModule`的实现类里定义.

如代码所示, 在一个RDD数据集上执行模型有如下几步:

**1.将模型广播到各个节点**

```scala
    val modelBroad = ModelBroadcast[T]().broadcast(dataset.sparkContext, model.evaluate())
```
这一句将模型拷贝到了每一个spark节点上, 让其都能访问到.

**2.将vMethods和一个能将数据集转为一个个batch的transformer广播到各个节点**

```scala
    val otherBroad = dataset.sparkContext.broadcast
    (
     vMethods, 
     SampleToMiniBatch(batchSize = totalBatch, partitionNum = Some(partitionNum))
     )
```
这里注意一下一个scala语法的坑, 事实上`broadcast`函数只能接受一个参数, 但是scala支持函数不带括号的调用语法,
比如`a.add b`等价于`a.add(b)`, 所以这里的参数其实是一个Tuple: `(vMethods, SampleToMiniBatch(...))`.

**3.在每个节点执行一遍模型然后收集结果**

代码就是这一堆:

```scala
    dataset.mapPartitions(partition => {
      val localModel = modelBroad.value()
      val localMethod = otherBroad.value._1.map(_.clone())
      val localTransformer = otherBroad.value._2.cloneTransformer()
      val miniBatch = localTransformer(partition)
      miniBatch.map(batch => {
        val output = localModel.forward(batch.getInput())
        localMethod.map(validation => {
          validation(output, batch.getTarget())
        })
      })
    }).reduce((left, right) => {
        left.zip(right).map { case (l, r) => l + r }
    }).zip(vMethods)
```

先是最顶层的`mapPartitions`, 简单, spark的机制是一个节点保存一个partition, 所以呢这个就是在每个节点执行一遍后面的那个函数`partition=>{...}`.

`partition`这个参数就是一个数据分区了.

继续看函数体, 前3句:

```scala
      val localModel = modelBroad.value()
      val localMethod = otherBroad.value._1.map(_.clone())
      val localTransformer = otherBroad.value._2.cloneTransformer()
```
前面说了在前2步广播了几个变量, 这里就是在slave上访问那几个变量, `localModel`是模型, `localMethod`是那个统计方法数组,
`localTransformer`就是把数据转成一个个batch的对象.

 然后就是调用这个`localTransformer`将数据集转成batch.
 
 后面的代码, 除了这一句:
 ```scala
val output = localModel.forward(batch.getInput())
```
是运行模型inference外, 其他都是在收集统计结果, 可以不必关注.

所以我们后面至于关注模型如何forward.

这个留在下一节 [forward](forward.md)详述.

 
 