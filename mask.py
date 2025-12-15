import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    input_ids = inputs["input_ids"][0]


    # Convert TensorFlow tensor to list if needed
    try:
        input_ids = input_ids.numpy().tolist()
    except AttributeError:
        pass


    for i, token_id in enumerate(input_ids):
        if token_id == mask_token_id:
            return i
    return None



def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    value = int(attention_score * 255)
    return (value, value, value)



def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    num_layers = len(attentions)

    for layer_index in range(num_layers):
        num_heads = len(attentions[layer_index])

        for head_index in range(num_heads):
            # Attention matrix for this layer and head
            attention_matrix = attentions[layer_index][head_index][0]

            generate_diagram(
                layer_index + 1,
                head_index + 1,
                tokens,
                attention_matrix
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")

# Testes das funções AI generated
if __name__ == "__main__":

    # Simulação de inputs do tokenizer
    inputs = {
        "input_ids": [[101, 2023, 2003, 103, 102]]
    }

    mask_token_id = 103  # ID padrão do [MASK] no BERT

    index = get_mask_token_index(mask_token_id, inputs)
    print("Mask index:", index)


    print(get_color_for_attention_score(0.0))   # (0, 0, 0)
    print(get_color_for_attention_score(0.5))   # (~127, ~127, ~127)
    print(get_color_for_attention_score(1.0))   # (255, 255, 255)

    tokens = ["[CLS]", "The", "cat", "sat", "[SEP]"]

    # Criar attentions falsas:
    # 2 layers, 2 heads, batch size 1, tokens x tokens
    attentions = []

    for _ in range(2):  # layers
        layer = []
        for _ in range(2):  # heads
            head = [
                [
                    [j / len(tokens) for j in range(len(tokens))]
                    for _ in range(len(tokens))
                ]
            ]  # batch dimension
            layer.append(head)
        attentions.append(layer)

    visualize_attentions(tokens, attentions)

    print("Diagramas de atenção gerados com sucesso.")
