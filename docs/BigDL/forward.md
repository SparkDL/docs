# forward

```scala
/**
* Takes an input object, and computes the corresponding output of the module. After a forward,
* the output state variable should have been updated to the new value.
*
* @param input input data
* @return output data
*/
final def forward(input: A): B = {
val before = System.nanoTime()
try {
  updateOutput(input)
} catch {
  case l: LayerException =>
    l.layerMsg = this.toString() + "/" + l.layerMsg
    throw l
  case e: Throwable =>
    throw new LayerException(this.toString(), e)
}
forwardTime += System.nanoTime() - before

output
}
```

前面已经提到过了, `forward`是`AbstractModule`抽象类定义好的函数, 做一些统计时间的工作,
实际执行计算的是`updateOutput`.

```scala
  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  def updateOutput(input: A): B
```


这个函数没有实现, 需要由子类来实现. 而我们也已经提到过, 我们构建最顶层的API, `Sequential`和`Graph`都是`AbstractModule`的实现类.

所以一个完整模型的`forward`过程, 最终执行的都是`Sequential.updateOutput`和`Graph.updateOutput`函数.

让我们逐一来分析一下.

在此之前可以先回顾一下`Module`的继承层次: [Module](Module.md).

## Sequential

首先来回顾一下`Sequential`构建模型的流程 [Sequential API](https://bigdl-project.github.io/master/#ProgrammingGuide/Model/Sequential/)
, 下面是用`Sequential`构建模型的2个例子.

### 多分支

结构:

```scala
Linear -> ReLU --> Linear -> ReLU ----> Add
               |-> Linear -> ReLU --|

```

代码:

```scala
val branch1 = Sequential().add(Linear(...)).add(ReLU())
val branch2 = Sequential().add(Linear(...)).add(ReLU())
val branches = ConcatTable().add(branch1).add(branch2)

val model = Sequential()
model.add(Linear(...))
model.add(ReLU())
model.add(branches)
model.add(CAddTable())
```

### 多个输入

```scala
Linear -> ReLU ----> Add
Linear -> ReLU --|
```

```scala
val model = Sequential()
val branches = ParallelTable()
val branch1 = Sequential().add(Linear(...)).add(ReLU())
val branch2 = Sequential().add(Linear(...)).add(ReLU())
branches.add(branch1).add(branch2)
model.add(branches).add(CAddTable())
```

### Sequential.updateOutput

可以发现, `Sequential`是按照线性的模式一层层的构建神经网络, 无论是怎样复杂的模型, 都可以放在容器里然后按照顺序连接起来, 
所以在执行的时候就按照顺序`forward`一遍就可以了.

看`udpateOutput`的代码:
 
```scala
@SerialVersionUID(5375403296928513267L)
class Sequential[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends DynamicContainer[Activity, Activity, T] {

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    var result = input.asInstanceOf[Activity]
    while (i < modules.length) {
      result = modules(i).forward(result)
      i += 1
    }

    this.output = result
    output
  }
}
```

非常之简单... 就是按顺序把每一层的模型都`forward`一遍.

## Graph

`Graph`的情况复杂一点, 其又分成`StaticGraph`和`DynamicGraph`, 暂时没用到`DynamicGraph`, 只看`StaticGraph`吧.

还是先看2个例子, 来自 [Grpah API](https://bigdl-project.github.io/master/#ProgrammingGuide/Model/Functional/):

### 多分支

结构:

```scala
Linear -> ReLU --> Linear -> ReLU ----> Add
               |-> Linear -> ReLU --|
```

代码:

```scala
val linear1 = Linear(...).inputs()
val relu1 = ReLU().inputs(linear1)
val linear2 = Linear(...).inputs(relu1)
val relu2 = ReLU().inputs(linear2)
val linear3 = Linear(...).inputs(relu1)
val relu3 = ReLU().inputs(linear3)
val add = CAddTable().inputs(relu2, relu3)
val model = Graph(Seq[linear1], Seq[add])
```

### 多个输入

```scala
Linear -> ReLU ----> Add
Linear -> ReLU --|
```

```scala
val linear1 = Linear(...).inputs()
val relu1 = ReLU().inputs(linear1)
val linear2 = Linear(...).inputs()
val relu2 = ReLU().inputs(linear2)
val add = CAddTable().inputs(relu1, relu2)
val model = Graph(Seq[linear1, linear2], Seq[add])
```

可以发现从输入到输出, 整个模型构成了一个有向图, 这个比`Sequential`的情况要更复杂一些.

来看`StaticGraph`的`updateOutput`代码.

### StaticGraph.updateOutput

```scala
override def updateOutput(input: Activity): Activity = {
    var i = 0
    while(i < forwardExecution.length) {
        val node = forwardExecution(i)
        val nodeInput = findInput(node, input)
        inputCache(i) = nodeInput
        node.element.forward(nodeInput)
        i += 1
    }
    
    output = dummyOutput.element.output
    output
}
```

看一下这个`forwardExecution`是怎么来的:

```scala
private val forwardExecution = forwardGraph.topologySort.reverse
```

细节不用管, 总之它就是把整个图拓扑排序一下, 将所有的模块按照宽度优先的顺序排好, 然后再依次`forward`即可.




