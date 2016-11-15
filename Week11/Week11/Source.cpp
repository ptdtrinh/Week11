#include <iostream>

class DoublyLayeredNN
{
public:
	float a0 = 0.0f;
	float b0 = 0.0f;
	float a1 = 0.0f;
	float b1 = 0.0f;

	float getY(const float& x_input)
	{
		return a1 * a0 * x_input + a1 * b0 + b1;
	}
};

const int num_data = 5;

int main()
{
	const float study_time_data[num_data] = { 0.1, 0.2, 0.3, 0.4, 0.5 };
	const float score_data[num_data] = { 4, 5, 6, 7, 8 };

	// input x is study time -> black box(AI) -> output y is score
	// linear hypothesis : y = a * x + b
	DoublyLayeredNN model;

	for (int tr = 0; tr < 100; tr++)
		for (int i = 0; i < num_data; i++)
		{
			// let's train our linear hypothesis to answer correctly!
			const float x_input = study_time_data[i];
			const float y_output = model.getY(x_input);
			const float y_target = score_data[i];
			const float error = y_output - y_target;
			// we can consider that our LH is trained well when error is 0 or small enough
			// we define squared error
			const float sqr_error = 0.5 * error * error; // always zero or positive

														 // we want to find good combination of a and b which minimizes sqr_error

														 // sqr_error = 0.5 * (a * x + b - y_target)^2
														 // d sqr_error / da = 2*0.5*(a * x + b - y_target) * x; 
														 // d sqr_error / db = 2*0.5*(a * x + b - y_target) * 1;

			const float dse_over_da1 = error * (-1.0) * (model.a0 * x_input + model.b0);
			const float dse_over_db1 = error * (-1.0);
			
			const float lr = 0.1; // small number
			model.a1 -= dse_over_da1 * lr;
			model.b1 -= dse_over_db1 * lr;

			const float dse_over_da0 = error * (-1.0) * model.a1 * x_input;
			const float dse_over_db0 = error * (-1.0) * model.a1;

			// need to find good a and b
			// we can update a and b to decrease squared error
			// this is the gradient descent method
			// learning rate
			
			model.a0 -= dse_over_da0 * lr;
			model.b0 -= dse_over_db0 * lr;

			//std::cout <<"x_input="<<x_input<<" y_target="<<y_target
			//	<<" y_output="<<y_output << " sqr_error = "<< sqr_error<< std::endl;
		}

	// trained hypothesis
	std::cout << "a1 = " << model.a1 <<" b1 = " <<model.b1
		      <<" a0 = " <<model.a0 <<" b0 = " <<model.b0<< std::endl;

	return 0;
}