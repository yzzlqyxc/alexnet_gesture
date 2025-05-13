CC = gcc
CFLAGS = -Wall -Wextra -O2

SRCS = main.c network.c get_jgp.c
OBJS = $(SRCS:.c=.o)
TARGET = main

# Default rule
all: $(TARGET)

# Link object files to create the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -ljpeg  -o $@ $^

# Compile .c files to .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Optional: Phony targets to avoid conflicts with files named 'clean', 'all', etc.
.PHONY: all clean
