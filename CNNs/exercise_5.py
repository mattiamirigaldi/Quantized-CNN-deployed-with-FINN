"""
In this last exercise you will port the CNN exported with Brevitas to FINN on custom accelerator implemented on a
ZCU104 development board. Due to limitations on the FINN version made available to the public by Xilinx and time
constraint of the special project, you will only port the fashion MNIST model to FPGA.

FINN uses PYNQ to deploy the bitstream to the FPGA, you can read more information on PYNQ here:
https://pynq.readthedocs.io/en/latest/index.html
More information on our specific development board can be found here.
https://pynq.readthedocs.io/en/latest/getting_started/zcu104_setup.html

For this exercise you should have already implemented, trained and exported the CNN model in FINN format.
The Vivado 2020.1 suite, FINN docker and ZCU104 development board that are required to complete this task have been
prepared and tested previously. You should use the remote server during all the design and deployment phases of this
exercise. Check the telegram channel for login instructions, each student has its own home directory and credentials.
The IP name of the pynq board is pynq-zcu104, IP address is 192.168.166.58. You can connect to this IP only while using
the university WiFi/Ethernet. The same thing applies to the server with all the tools and necessary computing resources.
Therefore, you can only complete this exercise while being at the university.
Since you will not have physical access to the server, you will need to connect with the following command from a linux
machine ssh -X <your username>@pc-lse-1861.polito.it in order to use GUI applications.
Otherwise, if you are on windows, you should use X2GO or Mobaxterm. With Mobaxterm you should be able to work also when
you are not on university ground, as it supports ssh tunnelling.

Assignment:
- Read the FINN documentation before writing any code or executing the notebooks. If any problem occurs during the
  execution of the cells inside the jupyter notebooks, it is probably because of some wrong configuration done with the
  environment variables. https://finn.readthedocs.io/en/latest/getting_started.html#quickstart
- Read and understand the FINN tutorials, you can launch them using the command "./run-docker.sh notebook" inside the
  FINN folder in /home/tools/FINN .
  A summary of what is done in the tutorials can be found here: https://finn.readthedocs.io/en/latest/tutorials.html
- For any problems, first check the FAQ https://finn.readthedocs.io/en/latest/faq.html, then the official GitHub
  repository discussion page https://github.com/Xilinx/finn/discussions. For Brevitas issue please refer to its own
  gitter https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
- After you have completed all tutorials, follow the instructions of /end2end_example/cybersecurity tutorials to deploy,
  validate and extract hardware metrics of your CNN model running on our ZCU104 development board.
  To complete this exercise you have to provide the hardware metrics of your model as presented at the end of the last
  tutorial, i.e., end2end_example/cybersecurity/3-build-accelerator-with-finn.ipynb
- In the final report you will have to include the code of the jupiter notebooks that you modified to deploy the CNN.
"""