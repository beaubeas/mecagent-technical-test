from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    CLIPProcessor,
)
import torch
from PIL import Image
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best


### STEP 1: DATA LOADING ###
def load_data():
    """Load the dataset from Hugging Face."""
    # Replace the dataset path if needed
    data = load_dataset(
        "CADCODER/GenCAD-Code", split=["train", "test"], cache_dir="./data_cache"
    )
    train_data, test_data = data[0], data[1]
    return train_data, test_data


### STEP 2: IMAGE PREPROCESSING ###
def preprocess_image(image, processor):
    """Process the PIL Image using CLIP Processor."""
    if isinstance(image, Image.Image):  # Check if the image is a PIL Image
        image_tensor = processor(images=image, return_tensors="pt").pixel_values
        return image_tensor  # Return processed image tensor
    raise ValueError("Expected a PIL Image for processing.")


### STEP 3: DATA PREPROCESSING ###
def preprocess_function(examples, tokenizer, processor):
    """
    Process dataset examples for Seq2Seq tasks (text and images as input).

    - Images: Processed using the processor into pixel embeddings.
    - Text (CadQuery codes): Tokenized and set as inputs/labels for the model.
    """
    # Process images
    processed_images = [preprocess_image(image, processor) for image in examples["image"]]

    # Tokenize CadQuery codes
    inputs = examples["prompt"]  # Use the input description/command as prompt
    targets = examples["cadquery"]  # Use CadQuery code as the output

    # Encode inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Tokenize targets and assign them to "labels"
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    # Add processed outputs to model_inputs
    model_inputs["labels"] = labels["input_ids"]  # Decoder labels
    model_inputs["pixel_values"] = torch.cat(processed_images, dim=0)  # Add processed images
    return model_inputs


def tokenize_data(train_data, test_data, tokenizer, processor):
    """Tokenize the train/test datasets."""
    tokenized_train = train_data.map(
        lambda x: preprocess_function(x, tokenizer, processor), batched=True
    )
    tokenized_test = test_data.map(
        lambda x: preprocess_function(x, tokenizer, processor), batched=True
    )
    return tokenized_train, tokenized_test


### STEP 4: TRAIN MODELS ###
def train_model(tokenized_train, tokenized_test, model_checkpoint, output_dir, epochs=3, batch_size=8):
    """
    Train a model with specific parameters.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=epochs,
        predict_with_generate=True,
        logging_dir=f"{output_dir}_logs",
    )

    # Define Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
    )

    trainer.train()
    return model, tokenizer


### STEP 5: EVALUATION ###
def generate_code(image, model, tokenizer, processor):
    """Generate CadQuery code for a single image."""
    # Process the input image
    image_tensor = preprocess_image(image, processor)

    # Pass the processed image to the model
    inputs = {
        "pixel_values": image_tensor,
    }
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_model(test_data, model, tokenizer, processor):
    """Evaluate VSR and IOU metrics for a given model."""
    predictions = []

    # Loop through the test data to generate predictions
    for example in test_data:
        try:
            predicted = generate_code(example["image"], model, tokenizer, processor)
            predictions.append(predicted)
        except Exception as e:
            print(f"Error processing example: {e}")
            predictions.append("")

    # Prepare predictions for VSR evaluation
    codes = {f"predicted_{i}": predicted for i, predicted in enumerate(predictions)}
    vsr = evaluate_syntax_rate_simple(codes)
    print("VSR:", vsr)

    # Evaluate IOU (if necessary, on a subset of samples)
    iou_scores = []
    for i in range(10):  # Test on 10 samples
        ground_truth = test_data[i]["cadquery"]
        predicted = predictions[i]
        iou = get_iou_best(predicted, ground_truth)
        iou_scores.append(iou)

    avg_iou = sum(iou_scores) / len(iou_scores)
    print("Average IOU:", avg_iou)

    return {"VSR": vsr, "Average IOU": avg_iou}


### MAIN FUNCTION ###
def main():
    """Main function to orchestrate training and evaluating both baseline and enhanced models."""
    # Load dataset
    print("Loading dataset...")
    train_data, test_data = load_data()

    # Image processor (for handling image feature extraction)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Baseline Model Training and Evaluation
    print("\n==== Training and Evaluating Baseline Model (T5-Small) ====")
    baseline_model_checkpoint = "t5-small"
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_checkpoint)
    tokenized_train, tokenized_test = tokenize_data(train_data, test_data, baseline_tokenizer, processor)
    baseline_model, baseline_tokenizer = train_model(
        tokenized_train, tokenized_test, baseline_model_checkpoint, output_dir="./baseline_model"
    )
    print("\nEvaluating Baseline Model...")
    baseline_metrics = evaluate_model(test_data, baseline_model, baseline_tokenizer, processor)

    # Enhanced Model Training and Evaluation
    print("\n==== Training and Evaluating Enhanced Model (T5-Base) ====")
    enhanced_model_checkpoint = "t5-base"
    enhanced_tokenizer = AutoTokenizer.from_pretrained(enhanced_model_checkpoint)
    tokenized_train, tokenized_test = tokenize_data(train_data, test_data, enhanced_tokenizer, processor)
    enhanced_model, enhanced_tokenizer = train_model(
        tokenized_train, tokenized_test, enhanced_model_checkpoint, output_dir="./enhanced_model"
    )
    print("\nEvaluating Enhanced Model...")
    enhanced_metrics = evaluate_model(test_data, enhanced_model, enhanced_tokenizer, processor)

    # Print Results
    print("\n=== Results Comparison ===")
    print(f"\nBaseline Model VSR: {baseline_metrics['VSR']}, IOU: {baseline_metrics['Average IOU']}")
    print(f"\nEnhanced Model VSR: {enhanced_metrics['VSR']}, IOU: {enhanced_metrics['Average IOU']}") 


if __name__ == "__main__":
    main()
