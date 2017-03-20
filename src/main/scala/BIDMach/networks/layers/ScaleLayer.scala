package BIDMach.networks.layers

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat,TMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks._

class ScaleLayer(override val net:Net, override val opts:ScaleNodeOpts = new ScaleNode) extends ModelLayer(net, opts, 2) {
  def initModelMat(nr:Int, nc:Int):Mat = {
    rand(nr, nc) - 0.5f;
  }
  
  override def forward = {
    val start = toc
    
    if (modelmats(imodel).asInstanceOf[AnyRef] == null) {
      // Multiplier
      modelmats(imodel) = convertMat(initModelMat(inputData.nrows, 1))
      // Bias
      modelmats(imodel + 1) = convertMat(initModelMat(inputData.nrows, 1))
      updatemats(imodel) = modelmats(imodel).zeros(modelmats(imodel).nrows, modelmats(imodel).ncols)
      updatemats(imodel + 1) = modelmats(imodel + 1).zeros(modelmats(imodel + 1).nrows, modelmats(imodel + 1).ncols)
    }
    
    // TODO we don't actually have to squash this do we
    output ~ inputData ∘ modelmats(imodel)
    if (opts.hasBias) {
      output ~ output + modelmats(imodel + 1)
    }
    
    forwardtime += toc - start
  }
  
  override def backward = {
    val start = toc
    
    inputDeriv ~ inputDeriv + modelmats(imodel)
    updatemats(imodel) ~ updatemats(imodel) + inputData
    if (opts.hasBias) {
      updatemats(imodel + 1) ~ updatemats(imodel + 1) + updatemats(imodel + 1).ones(updatemats(imodel + 1).nrows, updatemats(imodel + 1).ncols)
    }
    
    backwardtime += toc - start
  }

  override def toString = {
    "scale@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

trait ScaleNodeOpts extends ModelNodeOpts {
  var hasBias:Boolean = true
}

class ScaleNode extends Node with ScaleNodeOpts {
  override def clone:ScaleNode = copyTo(new ScaleNode).asInstanceOf[ScaleNode]
  
  override def create(net:Net):ScaleLayer = ScaleLayer(net, this)
  
  override def toString = {
    "scale@" + Integer.toHexString(hashCode() % 0x10000)
  }
}

object ScaleLayer {
  def apply(net:Net) = new ScaleLayer(net, new ScaleNode)
  
  def apply(net:Net, opts:ScaleNodeOpts) = new ScaleLayer(net, opts)
}
