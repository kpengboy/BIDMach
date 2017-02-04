package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks._

class BatchNormLayer(override val net:Net, override val opts:BatchNormNodeOpts = new BatchNormNode) extends ModelLayer(net, opts) {
  var epsilonMat:Mat = null
  
  def initModelMat(nr:Int, nc:Int):Mat = {
    // TODO what am I actually supposed to put here.
    rand(nr, nc) - 0.5f
  }

  override def forward = {
    val start = toc
    
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      modelmats(imodel) = convertMat(initModelMat(inputData.nrows, 2))
    }
    if (epsilonMat.asInstanceOf[AnyRef] == null) {
      epsilonMat = ones(1, 1)
      epsilonMat.set(opts.epsilon)
    }
    
    createOutput
    val means = mean(inputData, 2)
    val variances = variance(inputData, 2)
    val z = (inputData - means) / sqrt(variances + epsilonMat)
    output ~ modelmats(imodel).colslice(0, 1) ∘ z + modelmats(imodel).colslice(1, 2)
    clearDeriv

    forwardtime += toc - start
  }
  
  override def backward = {
    val start = toc
    
    // XXX what's the proper way to do this
    val means = mean(inputData, 2)
    val diffs = inputData - means
    val mdiffs = inputData / inputData.ncols - means
    val variancesPlusEps = variance(inputData, 2) + epsilonMat
    val thing = -inputData.ncols / sqrt(variancesPlusEps).t
    val matrow = (diffs ∘ mdiffs).t
    val diffmat = ones(inputData.ncols, 1) * matrow / variancesPlusEps.t ** 1.5
    diffmat ~ thing - diffmat
    diffmat ~ diffmat + mkdiag(variancesPlusEps ** -0.5)
    inputDeriv ~ inputDeriv + deriv ∘ diffmat
    
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
