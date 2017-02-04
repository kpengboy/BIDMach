package BIDMach.networks.layers

import BIDMach.networks._
import java.util.Arrays
import java.util.List

class ConvolutionLayer(override val net:Net, override val opts:ConvolutionNodeOpts = new ConvolutionNode) extends ModelLayer(net, opts) {

  override def toString = {
    "convolution@" + Integer.toHexString(hashCode() % 0x10000)
  }

}

trait ConvolutionNodeOpts extends ModelNodeOpts {
  var noutputs:Int = 0
  var hasBias:Boolean = true
  var pad:List[Integer] = null
  var kernel:List[Integer] = null
  var stride:List[Integer] = null
  var dilation:List[Integer] = Arrays.asList(1)
  var group:Int = 1
  var axis:Int = 1
  var forceND:Boolean = false
}

class ConvolutionNode extends Node with ConvolutionNodeOpts {

  override def clone:ConvolutionNode = copyTo(new ConvolutionNode).asInstanceOf[ConvolutionNode]
  
  override def create(net:Net):ConvolutionLayer = ConvolutionLayer(net, this)
  
  override def toString = {
    "convolution@" + Integer.toHexString(hashCode() % 0x10000)
  }

}

object ConvolutionLayer {
  
  def apply(net:Net) = new ConvolutionLayer(net, new ConvolutionNode)
  
  def apply(net:Net, opts:ConvolutionNodeOpts) = new ConvolutionLayer(net, opts)

}
