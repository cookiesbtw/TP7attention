import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# ==============================================================================
# CONFIGURATION AND CONSTANTS
# ==============================================================================

# Pre-trained masked language model - Uses BERT base model trained on uncased English text
MODEL = "bert-base-uncased"

# Number of predictions to generate - K=3 means the top 3 most likely tokens for the masked position
K = 3

# ==============================================================================
# CONSTANTS FOR GENERATING ATTENTION DIAGRAMS
# ==============================================================================

# Font object used for rendering text in attention diagrams
# Loads OpenSans-Regular font with size 28 from the assets folder
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)

# Size of each cell in the attention grid (in pixels)
# Each token interaction is represented by a square cell of this size
GRID_SIZE = 40

# Width of the left margin and top margin for token labels in diagrams (in pixels)
# Provides space to display token names alongside the attention matrix
PIXELS_PER_WORD = 200




def main():
    """
    Main entry point of the attention visualization program.
    
    This function orchestrates the entire workflow:
    1. Takes masked text input from the user (must include [MASK] token)
    2. Tokenizes the input text using the BERT tokenizer
    3. Validates that a mask token is present in the input
    4. Loads the pre-trained BERT masked language model
    5. Processes the input through the model to get predictions and attention scores
    6. Extracts the top K predictions for the masked position and prints them
    7. Generates attention visualization diagrams for all attention heads across all layers
    
    Raises:
        SystemExit: If the input text does not contain the [MASK] token
    """
    # Prompt user to input text containing the [MASK] token
    text = input("Text: ")

    # Initialize the BERT tokenizer and tokenize the input text
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    
    # Find the position (index) of the mask token in the tokenized input
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    
    # Validate that a mask token was found; exit if not present
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Load the pre-trained BERT model for masked language modeling
    # output_attentions=True ensures we get attention weights from all heads and layers
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Extract logits (unnormalized predictions) for the masked token position
    mask_token_logits = result.logits[0, mask_token_index]
    
    # Get the indices of the K tokens with the highest prediction scores
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    
    # Print K alternative completions by replacing the mask token with each top prediction
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Generate visualization diagrams for attention weights from all heads in all layers
    visualize_attentions(inputs.tokens(), result.attentions)



def get_mask_token_index(mask_token_id, inputs):
    """
    Locate and return the position (index) of the mask token in the tokenized input.
    
    This function searches through the input token IDs to find the mask token,
    which represents the position to be predicted by the masked language model.
    
    Args:
        mask_token_id (int): The token ID representing the [MASK] token (typically 103 for BERT)
        inputs (dict): Dictionary containing tokenized input, specifically 'input_ids' key
                      which holds a 2D tensor of token IDs (batch_size x sequence_length)
    
    Returns:
        int: The index position of the mask token in the sequence, or None if not found
    
    Logic Flow:
        1. Extract the input token IDs from the first (only) batch element
        2. Convert TensorFlow tensor to a Python list for iteration (if needed)
        3. Loop through each token ID and compare against the mask_token_id
        4. Return the index when a match is found
        5. Return None if the mask token is not present in the sequence
    """
    # Extract the input token IDs from the first batch element [0]
    input_ids = inputs["input_ids"][0]

    # Convert TensorFlow tensor to list if it hasn't been converted already
    # This allows us to iterate over token IDs as standard Python integers
    try:
        input_ids = input_ids.numpy().tolist()
    except AttributeError:
        # If input_ids is already a list, the conversion fails silently and we continue
        pass

    # Iterate through each token ID with its position index
    for i, token_id in enumerate(input_ids):
        # Check if the current token matches the mask token ID
        if token_id == mask_token_id:
            # Return the position of the mask token immediately when found
            return i
    
    # If we've searched the entire sequence without finding the mask token, return None
    return None



def get_color_for_attention_score(attention_score):
    """
    Convert an attention score to a grayscale color value.
    
    This function maps attention weights (normalized values between 0 and 1)
    to grayscale colors for visualization. Higher attention scores produce
    lighter colors (closer to white), while lower scores produce darker colors
    (closer to black).
    
    Args:
        attention_score (float): A normalized attention weight between 0.0 and 1.0,
                               where 1.0 represents maximum attention and 0.0 represents no attention
    
    Returns:
        tuple: An RGB color tuple (R, G, B) with three integers in range [0, 255]
               All three channels have the same value, creating a grayscale color
               Examples:
               - (0, 0, 0) = black (no attention)
               - (127, 127, 127) = medium gray (medium attention)
               - (255, 255, 255) = white (maximum attention)
    
    Logic Flow:
        1. Scale the attention score from [0, 1] range to [0, 255] range by multiplying by 255
        2. Convert the result to an integer (floor operation)
        3. Return an RGB tuple with the same value for all three channels (grayscale)
    """
    # Calculate the grayscale intensity value by scaling attention score to 0-255 range
    value = int(attention_score * 255)
    # Return RGB tuple with equal values for all channels to create grayscale color
    return (value, value, value)



def visualize_attentions(tokens, attentions):
    """
    Create and save visualization diagrams for all attention heads across all layers.
    
    This is the orchestration function that coordinates the generation of attention
    visualizations. It iterates through every attention head in every layer of the
    BERT model and creates a separate diagram for each one.
    
    Args:
        tokens (list): List of string tokens from the tokenized input
                      Example: ["[CLS]", "the", "cat", "[MASK]", "[SEP]"]
        attentions (tuple): Tuple of attention tensors from the model output
                           Structure: (num_layers,) where each layer contains
                           (batch_size, num_heads, seq_length, seq_length) tensor
    
    Logic Flow:
        1. Count the total number of transformer layers in the model
        2. Loop through each layer index (0 to num_layers-1)
        3. For each layer, count the number of attention heads
        4. Loop through each attention head index (0 to num_heads-1)
        5. Extract the attention weight matrix for the current layer and head
           - attentions[layer_index][head_index][0] 
           - [0] indexes into the batch dimension to get the first batch element
        6. Call generate_diagram() to create and save a visualization for that head
           - Pass layer number (1-indexed) and head number (1-indexed) for clearer file naming
    
    Note: The diagram filenames use 1-indexed layer and head numbers for readability
    """
    # Count total number of transformer layers by getting the length of the attentions tuple
    num_layers = len(attentions)

    # Iterate through each transformer layer
    for layer_index in range(num_layers):
        # Count the number of attention heads in this layer
        num_heads = len(attentions[layer_index])

        # Iterate through each attention head in the current layer
        for head_index in range(num_heads):
            # Extract the attention weight matrix for this specific layer and head
            # [0] accesses the first batch element (batch size is 1 in our case)
            # Result is a 2D tensor: (seq_length, seq_length) showing attention between all token pairs
            attention_matrix = attentions[layer_index][head_index][0]

            # Generate and save a diagram for this attention head
            # Using 1-indexed layer and head numbers (layer_index+1, head_index+1) for clarity in filenames
            generate_diagram(
                layer_index + 1,
                head_index + 1,
                tokens,
                attention_matrix
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate and save a single attention visualization diagram for one attention head.
    
    This function creates a 2D grid visualization where:
    - Each row represents attention FROM a token
    - Each column represents attention TO a token
    - Each cell's grayscale color indicates the attention weight (darker = less attention, lighter = more attention)
    - Tokens are labeled on both the left side (rows) and top side (columns) for reference
    
    Args:
        layer_number (int): The layer number (1-indexed) for use in the output filename
        head_number (int): The head number (1-indexed) for use in the output filename
        tokens (list): List of string tokens to display as labels
                      Example: ["[CLS]", "the", "cat", "[MASK]", "[SEP]"]
        attention_weights (tf.Tensor): 2D tensor of shape (num_tokens, num_tokens)
                                      containing normalized attention scores between 0 and 1
                                      attention_weights[i][j] = attention from token i to token j
    
    Logic Flow:
        1. Calculate the required image size based on number of tokens and grid dimensions
        2. Create a new transparent image with black background
        3. Render token labels vertically on the right side (column headers)
        4. Render token labels horizontally on the top side (row headers)
        5. Fill each cell with a grayscale color based on its attention weight:
           - Loop through all token pairs (i, j)
           - Calculate cell position based on GRID_SIZE and PIXELS_PER_WORD margins
           - Get grayscale color from attention weight using get_color_for_attention_score()
           - Draw rectangle for that cell
        6. Save the final image with filename: Attention_Layer{layer_number}_Head{head_number}.png
    
    Visual Layout:
        - Image margin (PIXELS_PER_WORD pixels): Space for token labels
        - Grid area: Square matrix of GRID_SIZE x GRID_SIZE cells
        - Right side labels: Tokens displayed vertically for each row
        - Top labels: Tokens displayed horizontally for each column
    """
    # Calculate total image size: margin space + grid space for all tokens
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    
    # Create a new image with RGBA color mode and black background
    # RGBA allows transparency; black (0,0,0) is used as the base color
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # =======================
    # DRAW TOKEN LABELS (COLUMNS - Vertical text on right side)
    # =======================
    for i, token in enumerate(tokens):
        # Create a separate transparent image for the rotated token label
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        
        # Draw the token text horizontally, positioned to be rotated later
        # Position: right side of image, row-aligned with grid cells
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        
        # Rotate the entire image 90 degrees to make text vertical
        token_image = token_image.rotate(90)
        
        # Paste the rotated token image onto the main image
        img.paste(token_image, mask=token_image)

    # =======================
    # DRAW TOKEN LABELS (ROWS - Horizontal text on left side)
    # =======================
    for i, token in enumerate(tokens):
        # Get the width of the token text to right-align it in the margin space
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        
        # Draw token label aligned to the left margin, one for each row
        # Positioned to align with each grid row
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # =======================
    # DRAW ATTENTION GRID
    # =======================
    # Iterate through all token pairs to create the attention matrix visualization
    for i in range(len(tokens)):
        # Calculate Y coordinate for this row (attention from token i)
        y = PIXELS_PER_WORD + i * GRID_SIZE
        
        for j in range(len(tokens)):
            # Calculate X coordinate for this column (attention to token j)
            x = PIXELS_PER_WORD + j * GRID_SIZE
            
            # Get the attention weight for this token pair
            # attention_weights[i][j] represents how much token i attends to token j
            attention_score = attention_weights[i][j]
            
            # Convert attention score to grayscale color
            # Higher scores = lighter gray; Lower scores = darker gray
            color = get_color_for_attention_score(attention_score)
            
            # Draw a rectangle cell for this attention weight
            # Rectangle coordinates: top-left (x, y) to bottom-right (x+GRID_SIZE, y+GRID_SIZE)
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save the final visualization image with a descriptive filename
    # Filename format: Attention_Layer{layer_number}_Head{head_number}.png
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")

# ==============================================================================
# TEST SECTION - VALIDATION AND DEMONSTRATION CODE
# ==============================================================================

if __name__ == "__main__":
    """
    This test section demonstrates and validates all functions without requiring
    actual BERT model inference. It creates mock data and tests each function
    independently to verify correctness of logic.
    """

    # ===== TEST 1: get_mask_token_index() =====
    # Create mock tokenized input simulating BERT tokenization
    # Structure: [CLS] + "This" + "is" + [MASK] + [SEP]
    # Token IDs: 101 (CLS), 2023 (This), 2003 (is), 103 (MASK), 102 (SEP)
    inputs = {
        "input_ids": [[101, 2023, 2003, 103, 102]]
    }

    # The [MASK] token ID in BERT is always 103
    mask_token_id = 103

    # Test the function: should return index 3 (0-indexed position of the mask token)
    index = get_mask_token_index(mask_token_id, inputs)
    print("Mask index:", index)


    # ===== TEST 2: get_color_for_attention_score() =====
    # Test with various attention scores to verify color mapping
    print(get_color_for_attention_score(0.0))   # Expected: (0, 0, 0) - Black (no attention)
    print(get_color_for_attention_score(0.5))   # Expected: (~127, ~127, ~127) - Medium gray
    print(get_color_for_attention_score(1.0))   # Expected: (255, 255, 255) - White (full attention)

    # ===== TEST 3: visualize_attentions() and generate_diagram() =====
    # Create a mock list of tokens as they would appear in a tokenized sentence
    tokens = ["[CLS]", "The", "cat", "sat", "[SEP]"]

    # Create mock attention tensors simulating model output
    # Structure: (num_layers, num_heads, batch_size, seq_len, seq_len)
    # We create 2 layers with 2 heads each for demonstration
    attentions = []

    for _ in range(2):  # Create 2 transformer layers
        layer = []
        for _ in range(2):  # Each layer has 2 attention heads
            # Create a mock attention matrix for one head
            # Attention values are between 0 and 1 (normalized)
            # We use a simple pattern: value = j / num_tokens (increases left to right)
            head = [
                [
                    # Create a sequence_length x sequence_length matrix
                    # All rows have the same pattern for demonstration
                    [j / len(tokens) for j in range(len(tokens))]
                    for _ in range(len(tokens))
                ]
            ]  # This inner list wraps it in batch dimension
            layer.append(head)
        attentions.append(layer)

    # Generate visualization diagrams for all attention heads
    # This will create 4 PNG files: Attention_Layer1_Head1.png, etc.
    visualize_attentions(tokens, attentions)

    # Print success message
    print("Diagramas de atenção gerados com sucesso.")

