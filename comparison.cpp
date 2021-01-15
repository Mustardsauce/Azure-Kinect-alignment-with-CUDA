#include <cstdio>
#include <chrono>

#include "cuda_k4a_align.h"
#include <opencv2/opencv.hpp>

cv::Mat get_mat_from_k4a(k4a::image& src, bool deep_copy=true);
cv::Mat k4a_get_mat(k4a_image_t& src, bool deep_copy = true);
void release_k4a_capture(k4a_capture_t& c);

void console_clear()
{
#if defined _WIN32
	system("cls");
	//clrscr(); // including header file : conio.h
#elif defined (__LINUX__) || defined(__gnu_linux__) || defined(__linux__)
	system("clear");
	//std::cout<< u8"\033[2J\033[1;1H"; //Using ANSI Escape Sequences 
#elif defined (__APPLE__)
	system("clear");
#endif
}

int main(void)
{			
	const float depth_scale = 0.001f;

	cuda_k4a_align color2depth;
	color2depth.set_align_status(true);

	cuda_k4a_align depth2color;
	depth2color.set_align_status(false);


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
	config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
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

	const k4a_transformation_t transformation = k4a_transformation_create(&calibration);

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
						
			printf("-------------------------------------\n");
			printf("color resoution : [%d, %d], depth resoution : [%d, %d]\n", color_calibration.resolution_width, color_calibration.resolution_height, depth_calibration.resolution_width, depth_calibration.resolution_height);

			{
				cudaEvent_t start, stop;
				float gpu_time = 0.0f;

				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);

				cv::Mat color2depth_alignment;

				color2depth_alignment = cv::Mat(cv::Size(depth_calibration.resolution_width, depth_calibration.resolution_height), CV_8UC3);
				color2depth.align_color_to_depth(color2depth_alignment.data, (ushort*)distorted_depthFrame.data, distorted_colorFrame.data, depth_scale, calibration);
				cv::imshow("color to depth (GPU)", color2depth_alignment);

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&gpu_time, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);

				printf("Time [color to depth (GPU)] : %f ms \n", gpu_time);
			}		

			{
				std::chrono::system_clock::time_point StartTime = std::chrono::system_clock::now();

				k4a_image_t transformed_color_image = NULL;
				if (K4A_RESULT_SUCCEEDED == 
					k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
					depth_calibration.resolution_width,
					depth_calibration.resolution_height,
					depth_calibration.resolution_width * 4 * (int)sizeof(uint8_t),
					&transformed_color_image))
				{		

					if (K4A_RESULT_SUCCEEDED == k4a_transformation_color_image_to_depth_camera(transformation,
						depth_image,
						color_image,
						transformed_color_image))
					{
						cv::Mat color2depth_alignment = k4a_get_mat(transformed_color_image);

						cv::imshow("color to depth (CPU)", color2depth_alignment);
					}
					
					k4a_image_release(transformed_color_image);
					
				}

				std::chrono::system_clock::time_point EndTime = std::chrono::system_clock::now();

				std::chrono::duration<double> DefaultSec = EndTime - StartTime;
				printf("Time [color to depth (CPU)] : %f ms \n", DefaultSec.count() * 1000.f);

			}

			{
				cudaEvent_t start, stop;
				float gpu_time = 0.0f;

				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);

				cv::Mat depth2color_alignment;
				depth2color_alignment = cv::Mat(cv::Size(color_calibration.resolution_width, color_calibration.resolution_height), CV_16UC1);
				depth2color.align_depth_to_color((ushort*)depth2color_alignment.data, (ushort*)distorted_depthFrame.data, depth_scale, calibration);

				cv::imshow("depth to color (GPU)", depth2color_alignment * depth_scale_for_visualization);

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&gpu_time, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);

				printf("Time [depth to color (GPU)] : %f ms \n", gpu_time);
			}

			{
				std::chrono::system_clock::time_point StartTime = std::chrono::system_clock::now();

				k4a_image_t transformed_depth_image = NULL;
				if (K4A_RESULT_SUCCEEDED == 
					k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
					color_calibration.resolution_width,
					color_calibration.resolution_height,
					color_calibration.resolution_width * (int)sizeof(uint16_t),
					&transformed_depth_image))
				{

					if (K4A_RESULT_SUCCEEDED ==
						k4a_transformation_depth_image_to_color_camera(transformation, depth_image, transformed_depth_image))
					{
						cv::Mat depth2color_alignment = k4a_get_mat(transformed_depth_image);

						cv::imshow("depth to color (CPU)", depth2color_alignment * depth_scale_for_visualization);
					}

					k4a_image_release(transformed_depth_image);

				}

				std::chrono::system_clock::time_point EndTime = std::chrono::system_clock::now();
				std::chrono::duration<double> DefaultSec = EndTime - StartTime;
				printf("Time [depth to color (CPU)] : %f ms \n", DefaultSec.count() * 1000.f);

			}

			cv::imshow("COLOR", distorted_colorFrame);
			cv::imshow("DEPTH", distorted_depthFrame * depth_scale_for_visualization);
			
			k4a_image_release(depth_image);
			k4a_image_release(color_image);
			cv::waitKey(10);
			console_clear();
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
