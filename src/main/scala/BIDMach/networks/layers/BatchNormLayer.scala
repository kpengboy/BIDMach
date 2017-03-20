package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks._

class BatchNormLayer(override val net:Net, override val opts:BatchNormNodeOpts = new BatchNormNode) extends ModelLayer(net, opts) {
  var epsilonMat:Mat = null
  var means:Mat = null
  var variances:Mat = null

  override def forward = {
    val start = toc
    
    // TODO implement test time logic
    if (epsilonMat.asInstanceOf[AnyRef] == null) {
      epsilonMat = ones(1, 1)
      epsilonMat.set(opts.epsilon)
    }
    
    createOutput
    means = mean(inputData, 2)
    variances = variance(inputData, 2)
    output ~ (inputData - means) / sqrt(variances + epsilonMat)
    clearDeriv

    forwardtime += toc - start
  }
  
  override def backward = {
    val start = toc
    
    // deriv = dl / dy
    // inputDeriv = dl / dx
    // updateMats(imodel) (or something) = dl / dW
    
    // TODO: implement test time logic
    
    val m = inputData.ncols
    val invStdev = 1 / sqrt(variances + epsilonMat)
    val devs = inputData - means
    val m2dldVar = sum(deriv ∘ devs, 2) * (invStdev ** 3)
    val dldMu = -sum(deriv, 2) * invStdev + m2dldVar * sum(devs, 2) / m
    inputDeriv ~ inputDeriv + (deriv * invStdev - m2dldVar * devs / m + dldMu / m)
    
    backwardtime += toc - start
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
  def apply(net:Net) = new BatchNormLayer(net)
  
  def apply(net:Net, opts:BatchNormNodeOpts) = new BatchNormLayer(net, opts)
}
