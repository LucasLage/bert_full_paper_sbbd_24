
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /src
ADD ./ /src

# Install the required packages
RUN apt-get update && apt-get install -y gcc && apt-get install -y g++

# Install tmux and zip
RUN apt-get install -y tmux && apt-get install -y zip

# Install the python required packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Jupyter notebook port
EXPOSE 8888

# Give permissions to the start.sh file
RUN chmod +x start.sh

# Run the start file
CMD ["sh", "start.sh"]
