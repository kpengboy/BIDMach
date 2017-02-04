package BIDMach.networks.layers

import BIDMach.networks._

class PoolingLayer(override val net:Net, override val opts:PoolingNodeOpts = new PoolingNode) extends ModelLayer(net, opts) {

  override def toString = {
    "pooling@" + Integer.toHexString(hashCode() % 0x10000)
  }

}

trait PoolingNodeOpts extends ModelNodeOpts {
  var padH:Int = 0
  var padW:Int = 0
  var kernelH:Int = 0
  var kernelW:Int = 0
  var strideH:Int = 0
  var strideW:Int = 0
  var globalPooling:Boolean = false
}

class PoolingNode extends Node with PoolingNodeOpts {

  override def clone:PoolingNode = copyTo(new PoolingNode).asInstanceOf[PoolingNode]
  
  override def create(net:Net) = PoolingLayer(net, this)
  
  override def toString = {
    "pooling@" + Integer.toHexString(hashCode() % 0x10000)
  }

}

object PoolingLayer {
  
  def apply(net:Net) = new PoolingLayer(net, new PoolingNode)
  
  def apply(net:Net, opts:PoolingNodeOpts) = new PoolingLayer(net, opts)

}