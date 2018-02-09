package BIDMach.extern

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,JSON,SMat,SDMat,TMat,FFilter,Filter,GFilter}
import BIDMat.MatFunctions._
import BIDMach._
import BIDMach.datasources.DataSource
import BIDMach.datasources.FileSource
import BIDMach.models.GLM
import BIDMach.networks.Net
import BIDMach.networks.layers._
import scala.collection.JavaConversions._
import scala.collection.generic.FilterMonadic
import scala.collection.mutable
import scala.language.implicitConversions
import scala.Option
import scala.util.control.Breaks._
import java.io.InputStream
import java.lang.IllegalArgumentException
import _root_.caffe.Caffe
import _root_.caffe.Caffe.LRNParameter.NormRegion
import _root_.caffe.Caffe.PoolingParameter.PoolMethod
import com.google.protobuf.{CodedInputStream,TextFormat}
import jcuda.jcudnn.cudnnPoolingMode

class CaffeModel private(net:Net, netParam:Caffe.NetParameterOrBuilder, _layers:Seq[CaffeLayer]) {
  import CaffeModel._
  
  private[extern] val layers = _layers

  def predictor(data:Mat, labels:Mat) = {
    val (nn, opts) = Net.predictor(net, data, labels)
    switchLayersToTest(nn.model.asInstanceOf[Net])
    (nn, opts)
  }
  
  def predictor(infn:String, outfn:String):(Learner, Net.FilePredOptions) = {
    predictor(List(FileSource.simpleEnum(infn,1,0)), List(FileSource.simpleEnum(outfn,1,0)));
  }

  def predictor(infn: String, inlb: String, outfn: String): (Learner, Net.FilePredOptions) = {
    predictor(List(FileSource.simpleEnum(infn, 1, 0), FileSource.simpleEnum(inlb, 1, 0)), List(FileSource.simpleEnum(outfn, 1, 0)));
  }

  def predLabels(infn: String, inlb: String): (Learner, Net.FilePredOptions) = {
    predictor(List(FileSource.simpleEnum(infn, 1, 0), FileSource.simpleEnum(inlb, 1, 0)), null);
  }
  
  def predictor(infiles:List[(Int)=>String], outfiles:List[(Int)=>String]):(Learner, Net.FilePredOptions) = {
    val (nn, opts) = Net.predictor(net, infiles, outfiles)
    switchLayersToTest(nn.model.asInstanceOf[Net])
    (nn, opts)
  }
  
  private def switchLayersToTest(newNet:Net) = {
    // It's assumed that model layers between train and test are the same
    val (_, testNodes) = parseProtobuf(netParam, Caffe.Phase.TEST, newNet)
    newNet.opts.nodeset = new NodeSet(testNodes.toArray)
  }
  
  def loadWeights(weightsFile:InputStream) = {
    val cis = CodedInputStream.newInstance(weightsFile)
    cis.setSizeLimit(1 << 30)
    val weightNetParam = Caffe.NetParameter.parseFrom(cis)
    
    // Build a map of names to layers for the weights
    val weightLayerForName = Map(weightNetParam.getLayerList().map(layer => (layer.getName(), layer)):_*)
    val modelMats = new mutable.ArrayBuffer[Mat]
    var i = 0
    while (i < layers.length) {
      val layer = layers(i)
      var incr = 1

      // If layer corresponds to a ModelNode, extract the model mats for this layer
      if (layer.inodeFirst != -1 && net.opts.nodeset(layer.inodeFirst).isInstanceOf[ModelNode]) {
        val weightLayer = weightLayerForName.get(layer.param.getName()) match {
          case Some(wl) => wl
          case None => throw new IllegalArgumentException(s"Layer ${layer.param.getName()} not found in weights file")
        }
        
        layer.param.getType() match {
          case "Convolution" => modelMats ++= getConvLayerMats(weightLayer, net)
          case "BatchNorm" => {
            if (i + 1 < layers.length && layers(i + 1).param.getType() == "Scale") {
              // We are loading data into a BatchNormScaleLayer
              assert(layer.inodeFirst == layers(i + 1).inodeFirst && layer.inodeLast == layers(i + 1).inodeLast)
              incr = 2
              val scaleWeightLayer = weightLayerForName.get(layers(i + 1).param.getName()) match {
                case Some(wl) => wl
                case None => throw new IllegalArgumentException(s"Layer ${layers(i + 1).param.getName()} not found in weights file")
              }
              modelMats ++= getScaleMats(scaleWeightLayer)
              // Theoretically this line should be outside the if statement, but at present the BatchNorm layer isn't a ModelLayer.
              modelMats ++= getBatchNormMats(weightLayer)
            }
          }
          case "Scale" => modelMats ++= getScaleMats(weightLayer)
          case "InnerProduct" => modelMats ++= getInnerProductLayerMats(weightLayer)
          case _ =>
        }
      }
      
      i += incr
    }
    net.setmodelmats(modelMats.toArray)
    net.opts.nmodelmats = modelMats.length
    net.refresh = false
  }
}

object CaffeModel {
  def loadModel(modelFile:Readable, net:Net) = {
    val caffeBuilder = Caffe.NetParameter.newBuilder()
    TextFormat.merge(modelFile, caffeBuilder)

    val (layers, nodes) = parseProtobuf(caffeBuilder, Caffe.Phase.TRAIN, net)
    net.opts.nodeset = new NodeSet(nodes.toArray)

    new CaffeModel(net, caffeBuilder, layers)
  }

  private def parseProtobuf(netParam:Caffe.NetParameterOrBuilder, phase:Caffe.Phase, net:Net) = {
    // Caffe only supports CrossCorrelation convolution
    net.opts.convType = Net.CrossCorrelation
    
    val layersForPhase = filterLayers(netParam.getLayerList(), phase)
    val layers = toposort(resolveLayerLinks(layersForPhase))

    // TODO: enforce NCHW if necessary
    // Translate every layer and build a mapping of blobs to layers feeding into them
    val nodes = new mutable.ArrayBuffer[Node]
    // SoftmaxOutputNode for categorical classification
    var softmaxOutputNode:SoftmaxOutputNode = null
    var hasAccuracy = false
    implicit def singleNode(node:Node):Array[Node] = Array(node)
    var i = 0
    while (i < layers.length) {
      val layer = layers(i)
      var incr = 1

      // Translate layer according to its layer type
      val newNodes:Array[Node] = layer.param.getType() match {
        case "Convolution" => translateConvolution(layer, net)
        case "Pooling" => translatePooling(layer)
        case "LRN" => translateLRN(layer)
        case "BatchNorm" => {
          val bnParam = layer.param.getBatchNormParam()
          if (i + 1 < layers.length && layers(i + 1).inputs.contains(layer) && layers(i + 1).param.getType() == "Scale") {
            // Combine this layer and the next into a single BatchNormScale node
            incr = 2
            val scaleParam = layers(i + 1).param.getScaleParam()
            val node = translateBatchNorm(layer, scaleParam)
            layers(i + 1).inodeFirst = nodes.length
            layers(i + 1).inodeLast = nodes.length
            node
          } else {
            translateBatchNorm(layer, null)
          }
        }

        case "MultinomialLogisticLoss" => {
          softmaxOutputNode = new SoftmaxOutputNode { lossType = SoftmaxOutputLayer.CaffeMultinomialLogisticLoss }
          softmaxOutputNode
        }
        case "SoftmaxWithLoss" => {
          softmaxOutputNode = new SoftmaxOutputNode
          softmaxOutputNode
        }
        case "EuclideanLoss" => new GLMNode { links = GLM.linear }
        case "HingeLoss" => {
          if (layer.param.getHingeLossParam().getNorm() != Caffe.HingeLossParameter.Norm.L1) {
            throw new UnsupportedOperationException("Only L1 loss is supported")
          }
          new GLMNode { links = GLM.svm }
        }
        case "Accuracy" => {
          hasAccuracy = true
          Array()
        }

        case "ReLU" => new RectNode
        case "Sigmoid" => new SigmoidNode
        case "TanH" => new TanhNode
        case "BNLL" => new SoftplusNode

        case "Data" => {
          val dataParam = layer.param.getDataParam()
          
          if (net.opts.isInstanceOf[DataSource.Opts]) {
            net.opts.asInstanceOf[DataSource.Opts].batchSize = dataParam.getBatchSize()
          }
          
          addTransformNodes(layer.param.getTransformParam(), new InputNode)
        }
        case "MemoryData" => new InputNode
        case "HDF5Data" => new InputNode

        case "InnerProduct" => {
          val ipp = layer.param.getInnerProductParam()
          new LinNode {
            outdim = ipp.getNumOutput()
            hasBias = ipp.getBiasTerm()
          }
        }
        case "Split" => new CopyNode
        case "Softmax" => new SoftmaxNode
        case "Dropout" => {
          val dropoutParam = layer.param.getDropoutParam()
          new DropoutNode { frac = dropoutParam.getDropoutRatio() }
        }
        // TODO: implement base, shift, scale for the following two
        case "Exp" => new ExpNode
        case "Log" => new LnNode
        case "Scale" => {
          val scaleParam = layer.param.getScaleParam()
          new ScaleNode { hasBias = scaleParam.getBiasTerm() }
        }
        // TODO: change once we implement all layer types
        case unknownType => throw new NotImplementedError("\"%s\" is not implemented yet" format unknownType)
      }
      
      // Set lr scaling and model mat for ModelNodes
      newNodes match {
        case Array(modelNode:ModelNode) => {
          if (layer.param.getParamCount() >= 1) {
            modelNode.lr_scale = layer.param.getParam(0).getLrMult()
            if (layer.param.getParamCount() >= 2) {
              modelNode.bias_scale = layer.param.getParam(1).getLrMult()
            }
          }
          if (layer.param.getParamList().filter(_.hasDecayMult()).nonEmpty) {
            Mat.consoleLogger.warning("The decay_mult option is not implemented")
          }
        }
        case _ =>
      }
      
      // Set accuracy score option if there was one
      if (hasAccuracy && (softmaxOutputNode ne null)) {
        softmaxOutputNode.scoreType = SoftmaxOutputLayer.AccuracyScore
      }
      
      layer.inodeFirst = nodes.length
      layer.inodeLast = nodes.length + newNodes.length - 1

      if (!newNodes.isEmpty) {
        for ((input, i) <- layer.inputs.zipWithIndex) {
          newNodes(0).inputs(i) = nodes(input.inodeLast)
        }
      }
      
      nodes ++= newNodes
      i += incr
    }

    (layers, nodes)
  }
  
  /** Filters out layers in the given {@code LayerParameter} sequence to only contain ones
   *  wanted for the given {@code phase}.
   */
  private def filterLayers(layerList:Seq[Caffe.LayerParameter], phase:Caffe.Phase) = {
    layerList.withFilter(layer => {
      check(!(layer.getIncludeCount() > 0 && layer.getExcludeCount() > 0), layer,
            "only include rules xor exclude rules can be specified")
      if (layer.getIncludeCount() > 0) {
        layer.getIncludeList().exists(netStateRule => stateMatchesRule(netStateRule, phase))
      } else {
        !layer.getExcludeList().exists(netStateRule => stateMatchesRule(netStateRule, phase))
      }
    })
  }
  
  private def stateMatchesRule(netStateRule:Caffe.NetStateRule, phase:Caffe.Phase) = {
    var matches = true
    for (fieldDesc <- netStateRule.getAllFields().keys) {
      fieldDesc.getName() match {
        case "phase" => matches &= (netStateRule.getPhase() == phase)
        case _ => println(s"Warning: net state rule ${fieldDesc.getName()} is not implemented")
      }
    }
    matches
  }
  
  /** Creates a list of {@code Layer} objects, setting their {@code lowers} and {@code uppers}
   *  attributes based on their links to blobs in the protobuf.
   */
  private def resolveLayerLinks(layers:FilterMonadic[Caffe.LayerParameter, Seq[Caffe.LayerParameter]]):Seq[CaffeLayer] = {
    val layerBuf = new mutable.ArrayBuffer[CaffeLayer]
    for (layerParam <- layers) {
      layerBuf += new CaffeLayer(layerParam)
    }
    
    // Make a table of top -> layers whose params have that top
    val layersWithTop = new mutable.HashMap[String,mutable.Buffer[CaffeLayer]]
    for (layer <- layerBuf) {
      for (t <- layer.param.getTopList()) {
        layersWithTop.getOrElseUpdate(t, new mutable.ArrayBuffer) += layer
      }
    }
    
    // Assign layer inputs based on which Caffe blobs each layer links to.
    // When several layers reuse the same blob in-place, order the layers in the order they were
    // specified in the prototxt.
    val blobIterIndices = new mutable.HashMap[String,Int]
    for (layer <- layerBuf) {
      // XXX this should account for multiple bottom blobs
      if (layer.param.getBottomCount() >= 1) {
        val bottom = layer.param.getBottom(0)
        if (layer.param.getTopList().contains(bottom)) {
          val j = blobIterIndices.getOrElse(bottom, 0)
          layer.inputs += layersWithTop(bottom)(j)
          blobIterIndices(bottom) = j + 1
        } else {
          layer.inputs += layersWithTop(bottom).last
        }
      }
    }
    
    layerBuf
  }
  
  /** Toposort the list of layers if necessary.
   *  It's highly unlikely this is out of order, but might as well sort if necessary.
   */
  private def toposort(layers:Seq[CaffeLayer]):Seq[CaffeLayer] = {
    // Check to see if it's already toposorted. If so, don't recreate the list.
    val seen = new mutable.HashSet[CaffeLayer]
    var ok = true
    breakable {
      for (layer <- layers) {
        for (input <- layer.inputs) {
          if (!seen.contains(input)) {
            ok = false
            break
          }
        }
        seen += layer
      }
    }

    if (ok) {
      layers
    } else {
      // Slow path: do the sort
      // Remember that input pointers point in the opposite direction of data flow. Hence, inputs are sinks.
      val sorted = new mutable.ArrayBuffer[CaffeLayer]
      val previsited = new mutable.HashSet[CaffeLayer]
      val postvisited = new mutable.HashSet[CaffeLayer]
      def visit(layer:CaffeLayer):Unit = {
        if (!postvisited.contains(layer)) {
          check(!previsited.contains(layer), layer.param, "Cycle detected")
          previsited += layer
          for (input <- layer.inputs) {
            visit(input)
          }
          postvisited += layer
          sorted += layer
        }
      }
      for (layer <- layers if (!postvisited.contains(layer))) {
        visit(layer)
      }
      sorted
    }
  }
  
  private def translateConvolution(layer:CaffeLayer, net:Net) = {
    val convParam = layer.param.getConvolutionParam()
    
    val convNode = new ConvNode {
      noutputs = convParam.getNumOutput()
      hasBias = convParam.getBiasTerm()
      if (convParam.hasPadW()) {
        pad = convParam.getPadW() \ convParam.getPadH()
      } else if (convParam.getPadCount() == 0) {
        pad = irow(0)
      } else {
        pad = irow(convParam.getPadList().map(_.intValue()).toList)
      }
      if (convParam.hasKernelW()) {
        kernel = convParam.getKernelW() \ convParam.getKernelH()
      } else {
        kernel = irow(convParam.getKernelSizeList().map(_.intValue()).toList)
      }
      if (convParam.hasStrideW()) {
        stride = convParam.getStrideW() \ convParam.getStrideH()
      } else if (convParam.getStrideCount() == 0) {
        stride = irow(1)
      } else {
        stride = irow(convParam.getStrideList().map(_.intValue()).toList)
      }
      dilation = irow(convParam.getDilationList().map(_.intValue()).toList)
      
      // BIDMach (currently) only supports xavier initialization
      if (convParam.getWeightFiller().getType() != "xavier") {
        throw new NotImplementedError("Only xavier initialization is currently implemented for convolution layers")
      }
    }

    Array[Node](convNode)
  }
  
  private def getConvLayerMats(layerParam:Caffe.LayerParameter, net:Net) = {
    val convParam = layerParam.getConvolutionParam()
    val modelMats = new mutable.ArrayBuffer[Mat]

    // TODO: avoid duplication with translateConvolution
    val hasBias = convParam.getBiasTerm()
    val stride0 = if (convParam.hasStrideW()) {
      convParam.getStrideW() 
    } else if (convParam.getStrideCount() == 0) {
      1
    } else {
      convParam.getStride(0)
    }
    val pad0 = if (convParam.hasPadW()) {
      convParam.getPadW() 
    } else if (convParam.getPadCount() == 0) {
      0
    } else {
      convParam.getPad(0)
    }
    
    if (!hasBias) {
      check(layerParam.getBlobsCount() == 1, layerParam, "convolution layer without bias needs 1 matrix")
    } else {
      check(layerParam.getBlobsCount() == 2, layerParam, "convolution layer with bias needs 2 matrices")
    }

    // TODO: avoid duplicating code with ConvLayer here
    val shape = layerParam.getBlobs(0).getShape().getDimList().map(_.intValue()).toArray
    val filter = FFilter2Ddn(shape(3), shape(2), shape(1), shape(0), stride0, pad0)
    // TODO: is this an abstraction barrier violation
    layerParam.getBlobs(0).getDataList().map(_.floatValue()).copyToArray(filter.data)
    modelMats += (if (net.opts.useGPU && Mat.hasCUDA > 0 && Mat.hasCUDNN) {
      val x = GFilter(filter)
      x.convType = Net.CrossCorrelation
      x.setTensorFormat(jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW);
      x
    } else {
      filter
    })
    
    if (hasBias) {
      val n = layerParam.getBlobs(1).getShape().getDim(0).toInt
      modelMats += blob2Mat(layerParam.getBlobs(1)).reshapeView(n, 1, 1, 1)
    } else {
      modelMats += null
    }

    modelMats
  }
  
  private def translatePooling(layer:CaffeLayer) = {
    val poolingParam = layer.param.getPoolingParam()
    new PoolingNode {
      if (poolingParam.hasPadH()) {
        pady = poolingParam.getPadH()
        padx = poolingParam.getPadW()
      } else {
        pady = poolingParam.getPad()
        padx = pady
      }
      
      if (poolingParam.hasKernelH()) {
        h = poolingParam.getKernelH()
        w = poolingParam.getKernelW()
      } else {
        h = poolingParam.getKernelSize()
        w = h
      }
      
      if (poolingParam.hasStrideH()) {
        stridey = poolingParam.getStrideH()
        stridex = poolingParam.getStrideW()
      } else {
        stridey = poolingParam.getStride()
        stridex = stridey
      }

      poolingMode = poolingParam.getPool() match {
        case PoolMethod.MAX => cudnnPoolingMode.CUDNN_POOLING_MAX
        case PoolMethod.AVE => cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        case PoolMethod.STOCHASTIC => throw new NotImplementedError("Stochastic pooling is not supported yet")
      }
    }
  }
  
  private def translateLRN(layer:CaffeLayer) = {
    val lrnParam = layer.param.getLrnParam()
    if (lrnParam.getNormRegion() == NormRegion.WITHIN_CHANNEL) {
      new LRNwithinNode {
        dim = lrnParam.getLocalSize()
        alpha = lrnParam.getAlpha()
        beta = lrnParam.getBeta()
      }
    } else {
      new LRNacrossNode {
        dim = lrnParam.getLocalSize()
        alpha = lrnParam.getAlpha()
        beta = lrnParam.getBeta()
        k = lrnParam.getK()
      }
    }
  }
  
  private def getInnerProductLayerMats(layerParam:Caffe.LayerParameter) = {
    if (!layerParam.getInnerProductParam().getBiasTerm()) {
      check(layerParam.getBlobsCount() == 1, layerParam, "linear layer without bias needs 1 matrix")
      Array(blob2MatTranspose(layerParam.getBlobs(0)), null)
    } else {
      check(layerParam.getBlobsCount() == 2, layerParam, "linear layer without bias needs 2 matrices")
      check(layerParam.getBlobs(0).getShape().getDim(0) == layerParam.getBlobs(1).getShape().getDim(0),
            layerParam, "weight and bias dimensions for linear layer don't agree")
      
      val outDim = layerParam.getBlobs(0).getShape().getDim(0).intValue()
      val weightMat = blob2MatTranspose(layerParam.getBlobs(0))
      val biasMat = blob2MatTranspose(layerParam.getBlobs(1)).reshapeView(outDim, 1)
      Array(weightMat, biasMat)
    }
  }
  
  private def translateBatchNorm(layer:CaffeLayer, scaleParam:Caffe.ScaleParameter) = {
    val bnParam = layer.param.getBatchNormParam()
    
    if (scaleParam ne null) {
      new BatchNormScaleNode {
        epsilon = bnParam.getEps()
        expAvgFactor = bnParam.getMovingAverageFraction()
        // It appears that Caffe always uses Spatial activations
        batchNormMode = BatchNormLayer.Spatial
        hasBias = scaleParam.getBiasTerm()
      }
    } else {
      new BatchNormNode {
        epsilon = bnParam.getEps()
        expAvgFactor = bnParam.getMovingAverageFraction()
        batchNormMode = BatchNormLayer.Spatial
      }
    }
  }
  
  private def getBatchNormMats(layerParam:Caffe.LayerParameter) = {
    check(layerParam.getBlobsCount() == 3, layerParam, "batch norm needs 2 matrices and scale factor")
    check(layerParam.getBlobs(2).getDataCount() > 0, layerParam, "batch norm layer doesn't have a scale factor")

    val c = layerParam.getBlobs(0).getShape().getDim(0).toInt
    check(c == layerParam.getBlobs(1).getShape().getDim(0).toInt, layerParam, "batch norm matrices aren't the same shape")
    val scale = {
      val rawScale = layerParam.getBlobs(2).getData(0)
      if (rawScale == 0) 0f else 1f / rawScale
    }
    val runningMeans = blob2Mat(layerParam.getBlobs(0)).reshapeView(c, 1, 1, 1)
    runningMeans ~ runningMeans * scale
    val runningVariances = blob2Mat(layerParam.getBlobs(1)).reshapeView(c, 1, 1, 1)
    runningVariances ~ runningVariances * scale
    Array(runningMeans, runningVariances)
  }
  
  private def getScaleMats(layerParam:Caffe.LayerParameter) = {
    val hasBias = layerParam.getScaleParam().hasBiasTerm()
    
    if (hasBias) {
      check(layerParam.getBlobsCount() == 2, layerParam, "scale layer with bias needs 2 matrices")
    } else {
      check(layerParam.getBlobsCount() == 1, layerParam, "scale layer without bias needs 1 matrix")
    }
    
    val c = layerParam.getBlobs(0).getShape().getDim(0).toInt
    val scaleMat = blob2Mat(layerParam.getBlobs(0)).reshapeView(c, 1, 1, 1)
    val biasMat = if (hasBias) {
      check(layerParam.getBlobs(1).getShape().getDim(0).toInt == c, layerParam, "scale layer matrices aren't the same shape")
      blob2Mat(layerParam.getBlobs(0)).reshapeView(c, 1, 1, 1)
    } else {
      zeros(c \ 1 \ 1 \ 1)
    }
    Array(scaleMat, biasMat)
  }
  
  private def addTransformNodes(transformParam:Caffe.TransformationParameter, subjectNode:Node) = {
    val newNodeList = new mutable.ListBuffer[Node]
    newNodeList += subjectNode

    if (transformParam.hasCropSize()) {
      val cropSize = transformParam.getCropSize()
      // TODO: use the correct dimensions
      val sizeMat = 0 \ cropSize \ cropSize \ 0
      // TODO: do I have to worry about offsets
      if (transformParam.getMirror()) {
        newNodeList += new CropMirrorNode {
          inputs(0) = newNodeList.last
          sizes = sizeMat
        }
      } else {
        newNodeList += new CropNode {
          inputs(0) = newNodeList.last
          sizes = sizeMat
        }
      }
    }

    // TODO: implement mean

    if (transformParam.hasScale()) {
      val constNode = new ConstantNode {
        value = transformParam.getScale()
        cache = true // TODO: verify
      }
      val mulNode = newNodeList.last ∘ constNode
      newNodeList += constNode
      newNodeList += mulNode
    }

    newNodeList.toArray
  }
  
  /** Converts the given blob into a Mat. Does not perform any transposition. */
  private def blob2Mat(blob:Caffe.BlobProto):Mat = {
    val dims = blob.getShape().getDimList().map(_.intValue()).toArray
    if (blob.getDoubleDataCount() > 0) {
      // TODO: should I bother with GDMat
      new DMat(dims, blob.getDoubleDataList().map(_.doubleValue()).toArray)
    } else {
      // TODO: should I bother with GFMat
      new FMat(dims, blob.getDataList().map(_.floatValue()).toArray)
    }
  }

  /** Converts the given blob into a Mat, transposing data from row-major order to column-major order. */
  private def blob2MatTranspose(blob:Caffe.BlobProto):Mat = {
    // We convert from row-major to column-major by creating a Mat with reversed dimensions,
    // loading it up with the row-major data, and then performing a deep transpose
    val dimList = blob.getShape().getDimList()
    val reverseDims = dimList.map(_.intValue()).reverse.toArray
    if (blob.getDoubleDataCount() > 0) {
      val data = blob.getDoubleDataList().map(_.doubleValue()).toArray
      // TODO: should I bother with GDMat
      new DMat(reverseDims, data).transpose((reverseDims.length - 1) to 0 by -1)
    } else {
      val data = blob.getDataList().map(_.floatValue()).toArray
      // TODO: should I bother with GFMat
      new FMat(reverseDims, data).transpose((reverseDims.length - 1) to 0 by -1)
    }
  }
  
  private def check(requirement:Boolean, layerParam:Caffe.LayerParameter, message: => Any) = {
    require(requirement, s"Layer ${layerParam.getName()}: ${message}")
  }
}

private class CaffeLayer(val param:Caffe.LayerParameter) {
  val inputs = new mutable.ArrayBuffer[CaffeLayer]
  var inodeFirst = -1
  var inodeLast = -1
}
