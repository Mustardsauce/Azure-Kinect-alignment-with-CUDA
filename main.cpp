#include <cstdio>

#include "cuda_k4a_align.h"
#include <opencv2/opencv.hpp>

cv::Mat get_mat_from_k4a(k4a::image& src, bool deep_copy=true);
cv::Mat k4a_get_mat(k4a_image_t& src, bool deep_copy = true);
void release_k4a_capture(k4a_capture_t& c);

void help__()
{
	printf("Input key (c) : Toggle the alignment mode (default - color to depth) \n");
}

int main(void)
{
	help__();

	cuda_k4a_align aligner;
	bool align_color_to_depth = true;
	const float depth_scale = 0.001f;
	aligner.set_align_status(align_color_to_depth);

	k4a_device_t device = NULL;
	uint32_t device_count = k4a_device_get_installed_count();

	const int depth_scale_for_visualization = 100;

	if (device_count == 0)
	{
		printf("No K4A devices found\n");
		return 0;
	}

	if (K4A_RESULT_SUCCEEDED !=
		k4a_device_open(K4A_DEVICE_DEFAULT, &device))
	{
		printf("Failed to open device\n");
		goto Exit;
	}

	k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	config.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
	config.color_resolution = K4A_COLOR_RESOLUTION_720P;
	config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
	config.camera_fps = K4A_FRAMES_PER_SECOND_30;

	if (K4A_RESULT_SUCCEEDED !=
		k4a_device_start_cameras(device, &config))
	{
		printf("Failed to start device\n");
		goto Exit;
	}

	k4a_calibration_t calibration;
	if (K4A_RESULT_SUCCEEDED !=
		k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration))
	{
		printf("Failed to get the calibration\n");
		k4a_device_close(device);
		goto Exit;
	}

	const auto& color_calibration = calibration.color_camera_calibration;
	const auto& depth_calibration = calibration.depth_camera_calibration;

	k4a_capture_t capture = NULL;
	while (true)
	{
		release_k4a_capture(capture);
		if (k4a_device_get_capture(device, &capture, 0) == K4A_WAIT_RESULT_SUCCEEDED)
		{
			k4a_image_t depth_image = k4a_capture_get_depth_image(capture);				
			k4a_image_t color_image = k4a_capture_get_color_image(capture);				

			if (color_image == NULL || depth_image == NULL)	
			{
				continue;
			}

			cv::Mat distorted_depthFrame = k4a_get_mat(depth_image);
			cv::Mat distorted_colorFrame = k4a_get_mat(color_image);

			k4a_image_release(depth_image);
			k4a_image_release(color_image);

			const bool status = aligner.get_align_status();

			cv::Mat alignment;
			if(status)
			{
				alignment = cv::Mat(cv::Size(depth_calibration.resolution_width, depth_calibration.resolution_height), CV_8UC3);
				aligner.align_color_to_depth(alignment.data, (ushort*)distorted_depthFrame.data, distorted_colorFrame.data, depth_scale, calibration);
				cv::imshow("ALIGN", alignment);
			}
			else
			{
				alignment = cv::Mat(cv::Size(color_calibration.resolution_width, color_calibration.resolution_height), CV_16UC1);
				aligner.align_depth_to_color((ushort*)alignment.data, (ushort*)distorted_depthFrame.data, depth_scale, calibration);			
				cv::imshow("ALIGN", alignment * depth_scale_for_visualization);
			}

			
			cv::imshow("COLOR", distorted_colorFrame);
			cv::imshow("DEPTH", distorted_depthFrame * depth_scale_for_visualization);

		}

		//
		const int key = cv::waitKey(10);

		if(key =='C' || key == 'c')
		{
			aligner.set_align_status(!aligner.get_align_status());

		}
	}


Exit:
	if (device != NULL)
	{
		k4a_device_close(device);
	}

	return 0;
}


cv::Mat k4a_get_mat(k4a_image_t& src, bool deep_copy)
{
	k4a_image_reference(src);
	return get_mat_from_k4a(k4a::image(src), deep_copy);
}

cv::Mat get_mat_from_k4a(k4a::image& src, bool deep_copy)
{
	assert(src.get_size() != 0);

	cv::Mat mat;
	const int32_t width = src.get_width_pixels();
	const int32_t height = src.get_height_pixels();

	const k4a_image_format_t format = src.get_format();
	switch (format)
	{
	case k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_MJPG:
	{
		// NOTE: this is slower than color formats.
		std::vector<uint8_t> buffer(src.get_buffer(), src.get_buffer() + src.get_size());
		mat = cv::imdecode(buffer, cv::IMREAD_COLOR);
		//cv::cvtColor(mat, mat, cv::COLOR_RGB2XYZ);
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_NV12:
	{
		cv::Mat nv12 = cv::Mat(height + height / 2, width, CV_8UC1, src.get_buffer()).clone();
		cv::cvtColor(nv12, mat, cv::COLOR_YUV2BGR);
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_YUY2:
	{
		cv::Mat yuy2 = cv::Mat(height, width, CV_8UC2, src.get_buffer()).clone();
		cv::cvtColor(yuy2, mat, cv::COLOR_YUV2BGR);
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_BGRA32:
	{
		mat = deep_copy ? cv::Mat(height, width, CV_8UC4, src.get_buffer()).clone()
			: cv::Mat(height, width, CV_8UC4, src.get_buffer());
		cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_DEPTH16:
	case k4a_image_format_t::K4A_IMAGE_FORMAT_IR16:
	{
		mat = deep_copy ? cv::Mat(height, width, CV_16UC1, reinterpret_cast<uint16_t*>(src.get_buffer())).clone()
			: cv::Mat(height, width, CV_16UC1, reinterpret_cast<uint16_t*>(src.get_buffer()));
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_CUSTOM8:
	{
		mat = cv::Mat(height, width, CV_8UC1, src.get_buffer()).clone();
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_CUSTOM:
	{
		// NOTE: This is opencv_viz module format (cv::viz::WCloud).
		const int16_t* buffer = reinterpret_cast<int16_t*>(src.get_buffer());
		mat = cv::Mat(height, width, CV_32FC3, cv::Vec3f::all(std::numeric_limits<float>::quiet_NaN()));
		mat.forEach<cv::Vec3f>(
			[&](cv::Vec3f& point, const int32_t* position) {
			const int32_t index = (position[0] * width + position[1]) * 3;
			point = cv::Vec3f(buffer[index + 0], buffer[index + 1], buffer[index + 2]);
		}
		);
		break;
	}
	default:
		throw k4a::error("Failed to convert this format!");
		break;
	}
	return mat;
}

void release_k4a_capture(k4a_capture_t& c)
{
	if (c != NULL)						
	{
		k4a_capture_release(c);			
		c = NULL;						
	}
}
