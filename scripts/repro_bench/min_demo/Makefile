.PHONY: demo rebuild clean

# Self-contained reproducibility demo. Run on any host with one or more
# NVIDIA GPUs and nvcc 12.x on PATH:
#
#     make demo

demo:
	@./demo.sh

rebuild:
	@./demo.sh --rebuild

clean:
	@rm -f /tmp/repro_demo_gpu*.log ../../../deploy/5dst_cuda/test_determinism
