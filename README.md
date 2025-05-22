
🎨 Neural Style Transfer

This project uses a neural network to blend the content of one image with the style of another. The result is a stylized version of the original photo that looks like it was painted by an artist. The script is based on the idea from the paper “A Neural Algorithm of Artistic Style” (Gatys et al.).

⸻

✨ Features
	•	Combines two images: one for content, one for style
	•	Uses VGG19 pretrained on ImageNet
	•	Preserves the structure of the original image while adding brush stroke patterns from the style image
	•	Customizable number of optimization steps and weights
	•	Works on CPU and GPU

⸻

🖼️ Screenshot


<img width="477" alt="Screenshot 2025-05-22 at 9 07 42 PM" src="https://github.com/user-attachments/assets/a73f1483-b767-45ca-b688-599772b28521" />


Stylized Output:

The final generated image (stylized_output.jpg) shows the content of content.jpg transformed using the style of style.jpg.
It was saved automatically after the model finished running for 300 steps.


⚙️ How It Works
	1.	Load content and style images
	2.	Pass them through the VGG19 network to extract features
	3.	Calculate content loss and style loss
	4.	Optimize a copy of the content image to minimize both losses
	5.	After several iterations, the output is saved with the combined visual effect


    💻 Installation

Make sure you have Python 3 installed, then install the following packages:
pip install torch torchvision pillow
python neural_style_transfer.py
Make sure you have the following files in the same directory:
	•	content.jpg
	•	style.jpg
