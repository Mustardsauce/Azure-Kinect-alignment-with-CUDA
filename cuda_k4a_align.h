#ifndef CUDA_K4A_ALIGN
#define CUDA_K4A_ALIGN

#include <k4a/k4a.hpp>   
#include <memory>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#ifdef _MSC_VER 
#pragma comment(lib, "cudart_static")
#endif

namespace cuda_align
{
	struct cuda_intrinsics
	{
		int width;
		int height;

		float fx;
		float fy;
		float cx;
		float cy;

		float k1;
		float k2;
		float k3;
		float k4;
		float k5;
		float k6;
		float codx; // center of distortion is set to 0 for Brown Conrady model
		float cody;
		float p1;
		float p2;
		float metric_radius;

		cuda_intrinsics()
		{
			width = height = fx = fy = cx = cy = k1 = k2 = k3 = k4 = k5 = k6 = codx = cody = p1 = p2 = metric_radius = 0.0;
		}

		cuda_intrinsics(const k4a_calibration_camera_t& in)
		{
			this->width = in.resolution_width;
			this->height = in.resolution_height;

			this->fx = in.intrinsics.parameters.param.fx;
			this->fy = in.intrinsics.parameters.param.fy;
			this->cx = in.intrinsics.parameters.param.cx;
			this->cy = in.intrinsics.parameters.param.cy;

			this->k1 = in.intrinsics.parameters.param.k1;
			this->k2 = in.intrinsics.parameters.param.k2;
			this->k3 = in.intrinsics.parameters.param.k3;
			this->k4 = in.intrinsics.parameters.param.k4;
			this->k5 = in.intrinsics.parameters.param.k5;
			this->k6 = in.intrinsics.parameters.param.k6;
			this->codx = in.intrinsics.parameters.param.codx; // center of distortion is set to 0 for Brown Conrady model
			this->cody = in.intrinsics.parameters.param.cody;
			this->p1 = in.intrinsics.parameters.param.p1;
			this->p2 = in.intrinsics.parameters.param.p2;

			this->metric_radius = in.metric_radius;

		}
	};

	struct cuda_extrinsics
	{
		float rotation[9];
		float translation[3];

		cuda_extrinsics()
		{
		}

		cuda_extrinsics(const k4a_calibration_extrinsics_t& ex)
		{
			memcpy(this->rotation, ex.rotation, sizeof(float) * 9);

			this->translation[0] = ex.translation[0] * 0.001;
			this->translation[1] = ex.translation[1] * 0.001;
			this->translation[2] = ex.translation[2] * 0.001;
		}
	};
}

class cuda_k4a_align
{

public:
	cuda_k4a_align() :
		d_depth_in(nullptr),
		d_color_in(nullptr),
		d_aligned_out(nullptr),
		d_pixel_map(nullptr),
		d_color_intrinsics(nullptr),
		d_depth_intrinsics(nullptr),
		d_depth_color_extrinsics(nullptr),
		align_status(true){}

	~cuda_k4a_align() { this->release(); }

	void align_color_to_depth(uint8_t* aligned_out, const uint16_t* depth_in, const uint8_t* color_in,
		float depth_scale, const k4a_calibration_t& calibration);

	void align_depth_to_color(uint16_t* aligned_out, const uint16_t* depth_in,
		float depth_scale, const k4a_calibration_t& calibration);

	const bool get_align_status() const			   { return this->align_status;   }
	void	   set_align_status(const bool status)
	{
		this->release();
		this->align_status = status; 
	}

private:
	std::shared_ptr<uint16_t>       d_depth_in;
	std::shared_ptr<unsigned char>  d_color_in;
	std::shared_ptr<unsigned char>  d_aligned_out;
	std::shared_ptr<int2>           d_pixel_map;

	std::shared_ptr<cuda_align::cuda_intrinsics> d_color_intrinsics;
	std::shared_ptr<cuda_align::cuda_intrinsics> d_depth_intrinsics;
	std::shared_ptr<cuda_align::cuda_extrinsics> d_depth_color_extrinsics;

	
	void release();

	bool align_status;

};



#endif
