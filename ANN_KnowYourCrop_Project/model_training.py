# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("🚀 STARTING CROP RECOMMENDATION MODEL TRAINING...")
print("=" * 60)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_and_explore_data():
    """Load and explore the dataset"""
    # Try to find the dataset in common locations
    possible_paths = [
        'enriched_crop_recommendation.csv',
        './data/enriched_crop_recommendation.csv',
        '../enriched_crop_recommendation.csv'
    ]

    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break

    if not dataset_path:
        print("❌ Dataset not found. Please ensure 'enriched_crop_recommendation.csv' is in the same folder.")
        return None

    print(f"📊 Loading dataset from: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📁 Dataset shape: {df.shape}")

        # Display basic information
        print("\n📋 Column Names:")
        print(df.columns.tolist())

        print(f"\n🔍 First 3 rows:")
        print(df.head(3))

        print(f"\n📈 Target variable distribution:")
        target_counts = df['label'].value_counts()
        print(target_counts)

        # Plot target distribution
        plt.figure(figsize=(12, 6))
        target_counts.plot(kind='bar')
        plt.title('Crop Distribution in Dataset')
        plt.xlabel('Crop Types')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('crop_distribution.png')
        print("📊 Saved crop distribution plot as 'crop_distribution.png'")

        return df

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def preprocess_data(df):
    """Preprocess the data: handle missing values, encode labels, scale features"""
    print("\n🔧 PREPROCESSING DATA...")

    # Create a copy for preprocessing
    df_processed = df.copy()

    # 1. Handle missing values
    print("1. Handling missing values...")
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    categorical_columns = df_processed.select_dtypes(include=['object']).columns

    # Fill numeric missing values with median
    for col in numeric_columns:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"   ✅ Filled missing values in {col} with median: {median_val:.2f}")

    # Fill categorical missing values with mode
    for col in categorical_columns:
        if df_processed[col].isnull().sum() > 0:
            mode_val = df_processed[col].mode()[0]
            df_processed[col].fillna(mode_val, inplace=True)
            print(f"   ✅ Filled missing values in {col} with mode: {mode_val}")

    # 2. Encode the target variable
    print("2. Encoding target variable...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_processed['label'])
    print(f"   ✅ Encoded {len(label_encoder.classes_)} crop types")
    print(f"   🌱 Crops: {list(label_encoder.classes_)}")

    # 3. Prepare features (exclude target column)
    print("3. Preparing features...")
    X = df_processed.drop('label', axis=1)

    # Handle categorical features by one-hot encoding
    categorical_features = X.select_dtypes(include=['object']).columns
    if len(categorical_features) > 0:
        print(f"   🔄 One-hot encoding categorical features: {list(categorical_features)}")
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    print(f"   ✅ Final feature count: {X.shape[1]}")
    print(f"   ✅ Feature names: {list(X.columns)}")

    return X, y, label_encoder, X.columns.tolist()


def build_model(input_dim, output_dim):
    """Build the neural network model"""
    print(f"\n🤖 BUILDING NEURAL NETWORK...")
    print(f"   Input dimensions: {input_dim}")
    print(f"   Output classes: {output_dim}")

    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        # Hidden layer 1
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        # Hidden layer 2
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        # Output layer
        Dense(output_dim, activation='softmax')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_and_evaluate_model(X_train, X_test, y_train, y_test, label_encoder):
    """Train and evaluate the model"""
    print("\n🚀 TRAINING MODEL...")

    # Build model
    input_dim = X_train.shape[1]
    output_dim = len(label_encoder.classes_)
    model = build_model(input_dim, output_dim)

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    print("⚖️ Class weights for handling imbalance:")
    for cls, weight in class_weight_dict.items():
        print(f"   {label_encoder.classes_[cls]}: {weight:.2f}")

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            verbose=1
        )
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluate the model
    print("\n📊 EVALUATING MODEL...")
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"🎯 Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"📉 Training Loss: {train_loss:.4f}")
    print(f"📉 Test Loss: {test_loss:.4f}")

    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification report
    print("\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("📊 Saved confusion matrix as 'confusion_matrix.png'")

    return model, history, test_accuracy


def save_model_and_artifacts(model, scaler, label_encoder, feature_names, test_accuracy):
    """Save the trained model and preprocessing artifacts"""
    print("\n💾 SAVING MODEL AND ARTIFACTS...")

    # Create directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)

    # Save model
    model_path = 'saved_models/crop_recommendation_model.h5'
    model.save(model_path)

    # Save preprocessing objects
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    joblib.dump(label_encoder, 'saved_models/label_encoder.pkl')
    joblib.dump(feature_names, 'saved_models/feature_names.pkl')

    # Save model info
    model_info = {
        'test_accuracy': float(test_accuracy),
        'num_classes': len(label_encoder.classes_),
        'num_features': len(feature_names),
        'classes': list(label_encoder.classes_),
        'features': feature_names
    }
    joblib.dump(model_info, 'saved_models/model_info.pkl')

    print("✅ Model and artifacts saved successfully!")
    print(f"📁 Model: saved_models/crop_recommendation_model.h5")
    print(f"📁 Scaler: saved_models/scaler.pkl")
    print(f"📁 Label Encoder: saved_models/label_encoder.pkl")
    print(f"📁 Feature Names: saved_models/feature_names.pkl")
    print(f"📁 Model Info: saved_models/model_info.pkl")


def test_sample_predictions(model, X_test, y_test, label_encoder, scaler, num_samples=5):
    """Test the model with sample predictions"""
    print(f"\n🧪 TESTING {num_samples} SAMPLE PREDICTIONS...")

    # Get random samples from test set
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    correct_predictions = 0

    for i, idx in enumerate(indices):
        sample = X_test[idx].reshape(1, -1)
        true_label = y_test[idx]

        # Make prediction
        prediction = model.predict(sample, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        true_crop = label_encoder.classes_[true_label]
        predicted_crop = label_encoder.classes_[predicted_class]

        status = "✅ CORRECT" if true_label == predicted_class else "❌ WRONG"
        if true_label == predicted_class:
            correct_predictions += 1

        print(f"Sample {i + 1}:")
        print(f"  True: {true_crop}")
        print(f"  Predicted: {predicted_crop}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Status: {status}")
        print()

    print(f"🎯 Sample Test Accuracy: {correct_predictions}/{num_samples} ({correct_predictions / num_samples:.1%})")


def main():
    """Main training pipeline"""
    # Step 1: Load and explore data
    df = load_and_explore_data()
    if df is None:
        return

    # Step 2: Preprocess data
    X, y, label_encoder, feature_names = preprocess_data(df)

    # Step 3: Scale features
    print("\n🔧 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Features scaled using StandardScaler")

    # Step 4: Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    print(f"✅ Features: {X_train.shape[1]}")

    # Step 5: Train and evaluate model
    model, history, test_accuracy = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, label_encoder
    )

    # Step 6: Save model and artifacts
    save_model_and_artifacts(model, scaler, label_encoder, feature_names, test_accuracy)

    # Step 7: Test sample predictions
    test_sample_predictions(model, X_test, y_test, label_encoder, scaler)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"🌱 Model can predict {len(label_encoder.classes_)} different crops")
    print(f"🔧 Features used: {len(feature_names)}")
    print(f"💾 Model saved in 'saved_models/' folder")
    print("🚀 You can now run the Flask app: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()