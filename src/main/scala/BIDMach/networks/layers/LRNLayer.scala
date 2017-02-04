package BIDMach.networks.layers

import BIDMach.networks._
import caffe.Caffe.LRNParameter.NormRegion

class LRNLayer(override val net:Net, override val opts:LRNNodeOpts = new LRNNode) extends ModelLayer(net, opts) {

  override def toString = {
    "lrn@"+Integer.toHexString(hashCode % 0x10000).toString
  }

}

trait LRNNodeOpts extends ModelNodeOpts {
  // TODO: do we actually want to double-define defaults here?
  var localSize:Int = 5
  var alpha:Float = 1f
  var beta:Float = 0.5f
  var normRegion:NormRegion = NormRegion.ACROSS_CHANNELS
  var k:Float = 1f
}

class LRNNode extends Node with LRNNodeOpts {

  override def clone:LRNNode = copyTo(new LRNNode).asInstanceOf[LRNNode]
  
  override def create(net:Net) = LRNLayer(net, this)
  
  override def toString = {
    "lrn@" + Integer.toHexString(hashCode() % 0x10000)
  }

}

object LRNLayer {
  
  def apply(net:Net) = new LRNLayer(net, new LRNNode)
  
  def apply(net:Net, opts:LRNNodeOpts) = new LRNLayer(net, opts)

}