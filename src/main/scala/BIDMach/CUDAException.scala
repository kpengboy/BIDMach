package BIDMach

// TODO: consider replacing with jcuda.CudaException
class CUDAException(val status:Int, val message:String = null, val cause:Throwable = null)
  extends RuntimeException("CUDA error " + status + (if (message != null) ": " + message else ""), cause) {
}
