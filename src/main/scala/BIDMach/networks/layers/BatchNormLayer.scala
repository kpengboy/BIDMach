package BIDMach.networks.layers

import jcuda._
import jcuda.runtime.JCuda._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.CUDAException
import BIDMach.networks._

class BatchNormLayer(override val net:Net, override val opts:BatchNormNodeOpts = new BatchNormNode) extends ModelLayer(net, opts) {
  import BatchNormLayer._
  
  var epsilonMat:Mat = null
  var means:Mat = null
  var variances:Mat = null
  
  var inputGMat:GMat = null
  var meansGMat:GMat = null
  var variancesGMat:GMat = null
  var scaleGMat:GMat = null
  var biasGMat:GMat = null

  override def forward = {
    val start = toc
    
    // TODO implement test time logic
    
    createOutput
    
    // TODO: don't assume cuDNN exists when CUDA exists (although we implicitly make this assumption elsewhere?)
    if (Mat.hasCUDA > 0) {
      forwardCUDA
    } else {
      forwardCPU
    }
    
    clearDeriv
    
    forwardtime += toc - start
  }
  
  // TODO: enable exceptions instead?
  def forwardCUDA = {
    var xDesc:cudnnTensorDescriptor = null
    var yDesc:cudnnTensorDescriptor = null
    var scaleBiasMeanVarDesc:cudnnTensorDescriptor = null
    // TODO: specify the GUID stuff
    // TODO how much of this can we reuse
    inputGMat = GMat.newOrCheckGMat(inputData.dims, inputData)
    val outputGMat = GMat.newOrCheckGMat(output.dims, output)
    meansGMat = GMat.make(Array(inputData.dims(0), inputData.dims(2), inputData.dims(1)))
    variancesGMat = GMat.make(Array(inputData.dims(0), inputData.dims(2), inputData.dims(1)))
    scaleGMat = GMat.ones(irow(1, inputData.dims(0), 1, 1))
    biasGMat = GMat.zeros(irow(1, inputData.dims(0), 1, 1))
    
    try {
      xDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        xDesc = null
        throw new OutOfMemoryError()
      }
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, TENSOR_FORMAT, DATA_TYPE, inputData.dims(3), inputData.dims(0), inputData.dims(2), inputData.dims(1))
      if (xSetStatus > 0) {
        throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm forward, bad stride?")
      }
      
      yDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(yDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        yDesc = null
        throw new OutOfMemoryError()
      }
      val ySetStatus = cudnnSetTensor4dDescriptor(yDesc, TENSOR_FORMAT, DATA_TYPE, inputData.dims(3), inputData.dims(0), inputData.dims(2), inputData.dims(1))
      if (ySetStatus > 0) {
        throw new CUDAException(ySetStatus, "Error creating y tensor for batch norm forward, bad stride?")
      }
      
      scaleBiasMeanVarDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(scaleBiasMeanVarDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        scaleBiasMeanVarDesc = null
        throw new OutOfMemoryError()
      }
      val sbmvDeriveStatus = cudnnDeriveBNTensorDescriptor(scaleBiasMeanVarDesc, xDesc, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL)
      if (sbmvDeriveStatus == cudnnStatus.CUDNN_STATUS_BAD_PARAM) {
        throw new CUDAException(sbmvDeriveStatus, "Error creating scale/bias/mean/var tensor for batch norm forward, bad stride?")
      }

      var err = cudnnBatchNormalizationForwardTraining(getHandle, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL,
        ONE, ZERO, xDesc, inputGMat.pdata, yDesc, outputGMat.pdata, scaleBiasMeanVarDesc, scaleGMat.pdata,
        biasGMat.pdata, 1.0, null, null, opts.epsilon, meansGMat.pdata, variancesGMat.pdata)
      cudaDeviceSynchronize()
      if (err == 0) {
        err = cudaGetLastError()
      }
      if (err > 0) {
        throw new CUDAException(err, "Error in CUDNN forward batch normalization: " + cudaGetErrorString(err))
      }
          
    } finally {
      if (scaleBiasMeanVarDesc != null) {
        cudnnDestroyTensorDescriptor(scaleBiasMeanVarDesc)
      }
      if (yDesc != null) {
        cudnnDestroyTensorDescriptor(yDesc)
      }
      if (xDesc != null) {
        cudnnDestroyTensorDescriptor(xDesc)
      }
    }
  }
  
  def forwardCPU = {
    if (epsilonMat.asInstanceOf[AnyRef] == null) {
      epsilonMat = ones(1, 1)
      epsilonMat.set(opts.epsilon)
    }
    
    means = mean(inputData, 2)
    variances = variance(inputData, 2)
    output ~ (inputData - means) / sqrt(variances + epsilonMat)
  }
  
  override def backward = {
    val start = toc
    
    // deriv = dl / dy
    // inputDeriv = dl / dx
    // updateMats(imodel) (or something) = dl / dW
    
    // TODO: implement test time logic
    
    if (Mat.hasCUDA > 0) {
      backwardCUDA
    } else {
      backwardCPU
    }
    
    backwardtime += toc - start
  }
  
  def backwardCUDA = {
    var xDesc:cudnnTensorDescriptor = null
    var dyDesc:cudnnTensorDescriptor = null
    var dxDesc:cudnnTensorDescriptor = null
    var scaleBiasDiffDesc:cudnnTensorDescriptor = null
    val derivGMat = GMat.newOrCheckGMat(deriv.dims, deriv)
    val inputDerivGMat = GMat.newOrCheckGMat(inputDeriv.dims, inputDeriv)
    val scaleDiffDummy = GMat.make(Array(1, inputData.dims(0), 1, 1))
    val biasDiffDummy = GMat.make(Array(1, inputData.dims(0), 1, 1))
    
    try {
      // TODO: try and avoid duplication of this
      xDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(xDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        xDesc = null
        throw new OutOfMemoryError()
      }
      val xSetStatus = cudnnSetTensor4dDescriptor(xDesc, TENSOR_FORMAT, DATA_TYPE, inputData.dims(3), inputData.dims(0), inputData.dims(2), inputData.dims(1))
      if (xSetStatus > 0) {
        throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward, bad stride?")
      }
      
      dyDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(dyDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        dyDesc = null
        throw new OutOfMemoryError()
      }
      val dySetStatus = cudnnSetTensor4dDescriptor(dyDesc, TENSOR_FORMAT, DATA_TYPE, inputData.dims(3), inputData.dims(0), inputData.dims(2), inputData.dims(1))
      if (dySetStatus > 0) {
        throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward, bad stride?")
      }
      
      dxDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(dxDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        dxDesc = null
        throw new OutOfMemoryError()
      }
      val dxSetStatus = cudnnSetTensor4dDescriptor(dxDesc, TENSOR_FORMAT, DATA_TYPE, inputData.dims(3), inputData.dims(0), inputData.dims(2), inputData.dims(1))
      if (dxSetStatus > 0) {
        throw new CUDAException(xSetStatus, "Error creating x tensor for batch norm backward, bad stride?")
      }
      
      scaleBiasDiffDesc = new cudnnTensorDescriptor()
      if (cudnnCreateTensorDescriptor(scaleBiasDiffDesc) == cudnnStatus.CUDNN_STATUS_ALLOC_FAILED) {
        scaleBiasDiffDesc = null
        throw new OutOfMemoryError()
      }
      val sbmvDeriveStatus = cudnnDeriveBNTensorDescriptor(scaleBiasDiffDesc, xDesc, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL)
      if (sbmvDeriveStatus == cudnnStatus.CUDNN_STATUS_BAD_PARAM) {
        throw new CUDAException(sbmvDeriveStatus, "Error creating scale/bias diff tensor for batch norm backward, bad stride?")
      }
      
      cudnnBatchNormalizationBackward(BatchNormLayer.getHandle, cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL,
          ONE, ZERO, ONE, ZERO, xDesc, inputGMat.pdata, dyDesc, derivGMat.pdata, dxDesc, inputDerivGMat.pdata,
          scaleBiasDiffDesc, scaleDiffDummy.pdata, scaleDiffDummy.pdata, biasDiffDummy.pdata, opts.epsilon,
          meansGMat.pdata, variancesGMat.pdata)

    } finally {
      if (xDesc != null) {
        cudnnDestroyTensorDescriptor(xDesc)
      }
    }
  }
  
  def backwardCPU = {
    val m = inputData.ncols
    val invStdev = 1 / sqrt(variances + epsilonMat)
    val devs = inputData - means
    val m2dldVar = sum(deriv ∘ devs, 2) * (invStdev ** 3)
    val dldMu = -sum(deriv, 2) * invStdev + m2dldVar * sum(devs, 2) / m
    inputDeriv ~ inputDeriv + (deriv * invStdev - m2dldVar * devs / m + dldMu / m)
  }
  
  override def toString = {
    "batchnorm@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

trait BatchNormNodeOpts extends ModelNodeOpts {
  var epsilon:Float = 1e-5f
}

class BatchNormNode extends Node with BatchNormNodeOpts {
  override def clone:BatchNormNode = copyTo(new BatchNormNode).asInstanceOf[BatchNormNode]

  override def create(net:Net) = BatchNormLayer(net, this)
  
  override def toString = {
    "batchnorm@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

object BatchNormLayer {
  // TODO: is this the right tensor format
  val TENSOR_FORMAT = cudnnTensorFormat.CUDNN_TENSOR_NCHW
  val DATA_TYPE = cudnnDataType.CUDNN_DATA_FLOAT
  
  val ONE = Pointer.to(Array(1.0f))
  val ZERO = Pointer.to(Array(0.0f))

  var cudnnContexts:Array[cudnnHandle] = null
  var cudnnContextsInitialized = false

  def apply(net:Net) = new BatchNormLayer(net)
  
  def apply(net:Net, opts:BatchNormNodeOpts) = new BatchNormLayer(net, opts)

  def initHandles = {
    BatchNormLayer.synchronized {
      if (!cudnnContextsInitialized) {
        val thisGPU = getGPU
        val nGPUs = Mat.hasCUDA
        cudnnContexts = new Array[cudnnHandle](nGPUs)
        for (i <- 0 until nGPUs) {
          setGPU(i)
          cudnnContexts(i) = new cudnnHandle()
          val err = cudnnCreate(cudnnContexts(i));
          if (err != 0) throw new CUDAException(err, "Cudnn initialization error on GPU %d" format i);
        }
        setGPU(thisGPU)
        cudnnContextsInitialized = true
      }
    }
  }

  def getHandle = {
    if (!cudnnContextsInitialized) initHandles
    cudnnContexts(getGPU)
  }
}
