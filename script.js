const terms = [
  { term: 'Algorithm', category: 'Machine Learning', level: 'Beginner', definition: 'A set of rules a computer follows to solve a task.', learnMore: 'In AI, algorithms define how models learn patterns and make predictions from data.' },
  { term: 'Model', category: 'Machine Learning', level: 'Beginner', definition: 'A trained mathematical function that maps inputs to outputs.', learnMore: 'Model quality depends on data quality, architecture choices, and evaluation rigor.' },
  { term: 'Training Data', category: 'Machine Learning', level: 'Beginner', definition: 'Examples used to teach a model during training.', learnMore: 'Representative training data helps models generalize better in real-world use.' },
  { term: 'Overfitting', category: 'Machine Learning', level: 'Intermediate', definition: 'When a model memorizes training data and fails on new examples.', learnMore: 'Regularization, early stopping, and better validation reduce overfitting.' },
  { term: 'Underfitting', category: 'Machine Learning', level: 'Intermediate', definition: 'When a model is too simple to learn meaningful patterns.', learnMore: 'Adding features, capacity, or training time can reduce underfitting.' },
  { term: 'Hyperparameter', category: 'Machine Learning', level: 'Intermediate', definition: 'A training setting chosen before learning starts.', learnMore: 'Common hyperparameters include learning rate, batch size, and regularization strength.' },
  { term: 'Gradient Descent', category: 'Machine Learning', level: 'Intermediate', definition: 'Optimization method that iteratively lowers model error.', learnMore: 'It updates parameters in the direction that minimizes the loss function.' },
  { term: 'Neural Network', category: 'Machine Learning', level: 'Beginner', definition: 'A layered model composed of interconnected artificial neurons.', learnMore: 'Deep neural networks can learn complex nonlinear patterns from large datasets.' },
  { term: 'Transformer', category: 'Generative AI', level: 'Advanced', definition: 'A sequence model architecture based on self-attention.', learnMore: 'Transformers power modern LLMs because they scale efficiently with compute and data.' },
  { term: 'Attention Mechanism', category: 'Generative AI', level: 'Advanced', definition: 'A method that helps models focus on relevant input parts.', learnMore: 'Self-attention captures token relationships across long contexts.' },
  { term: 'Large Language Model (LLM)', category: 'Generative AI', level: 'Beginner', definition: 'A very large transformer trained to predict and generate text.', learnMore: 'LLMs can perform summarization, reasoning, coding, and Q&A with prompts.' },
  { term: 'Prompt Engineering', category: 'Generative AI', level: 'Intermediate', definition: 'Designing prompts to improve model output quality and reliability.', learnMore: 'Structured prompts, constraints, and examples can improve consistency.' },
  { term: 'Fine-Tuning', category: 'Generative AI', level: 'Advanced', definition: 'Continuing training on domain data to specialize a model.', learnMore: 'Fine-tuning adapts behavior for tasks like legal drafting or medical coding.' },
  { term: 'Embeddings', category: 'NLP', level: 'Intermediate', definition: 'Vector representations that encode semantic meaning.', learnMore: 'Similar concepts are located closer in embedding space.' },
  { term: 'Tokenization', category: 'NLP', level: 'Beginner', definition: 'Splitting text into tokens for model processing.', learnMore: 'Tokenization strongly affects context length, cost, and multilingual behavior.' },
  { term: 'Named Entity Recognition (NER)', category: 'NLP', level: 'Intermediate', definition: 'Extracting entities like people, places, and organizations from text.', learnMore: 'NER supports information extraction in finance, healthcare, and legal systems.' },
  { term: 'Sentiment Analysis', category: 'NLP', level: 'Beginner', definition: 'Classifying emotional tone in text.', learnMore: 'It is widely used for brand monitoring and customer feedback analysis.' },
  { term: 'BLEU Score', category: 'NLP', level: 'Advanced', definition: 'A metric for machine translation quality using n-gram overlap.', learnMore: 'BLEU is useful but often combined with human evaluation for semantic adequacy.' },
  { term: 'Convolutional Neural Network (CNN)', category: 'Computer Vision', level: 'Intermediate', definition: 'A network that uses convolution filters to process image patterns.', learnMore: 'CNNs are strong for classification, detection, and segmentation tasks.' },
  { term: 'Object Detection', category: 'Computer Vision', level: 'Intermediate', definition: 'Finding and localizing objects in images or video.', learnMore: 'Detection models return labels plus bounding boxes for each object.' },
  { term: 'Image Segmentation', category: 'Computer Vision', level: 'Advanced', definition: 'Assigning class labels to each pixel in an image.', learnMore: 'Segmentation is critical for medical imaging and autonomous systems.' },
  { term: 'Transfer Learning', category: 'Machine Learning', level: 'Intermediate', definition: 'Reusing a pretrained model for a related new task.', learnMore: 'It reduces data and compute needs in low-resource settings.' },
  { term: 'Reinforcement Learning', category: 'Machine Learning', level: 'Advanced', definition: 'Learning through rewards by interacting with an environment.', learnMore: 'RL agents optimize long-term return using exploration and policy updates.' },
  { term: 'Hallucination', category: 'Generative AI', level: 'Intermediate', definition: 'A confident but incorrect model-generated statement.', learnMore: 'Grounding with retrieval and verification reduces hallucinations.' },
  { term: 'Retrieval-Augmented Generation (RAG)', category: 'Generative AI', level: 'Advanced', definition: 'Combining retrieval with generation for grounded responses.', learnMore: 'RAG injects relevant documents into context before generation.' },
  { term: 'Bias', category: 'Ethics & Safety', level: 'Beginner', definition: 'Systematic unfairness in model outputs.', learnMore: 'Bias can come from skewed data, labeling choices, or objective design.' },
  { term: 'Fairness', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Designing AI systems to avoid unfair treatment of groups.', learnMore: 'Fairness metrics differ and may require tradeoff decisions.' },
  { term: 'Explainability', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Making model decisions understandable to humans.', learnMore: 'Explainability builds trust and helps with debugging and compliance.' },
  { term: 'Adversarial Attack', category: 'Ethics & Safety', level: 'Advanced', definition: 'Crafted input designed to fool a model.', learnMore: 'Robust training and monitoring help defend against adversarial examples.' },
  { term: 'Alignment', category: 'Ethics & Safety', level: 'Advanced', definition: 'Keeping AI behavior consistent with human intentions and values.', learnMore: 'Alignment includes instruction following, harmlessness, and truthfulness.' },
  { term: 'Data Drift', category: 'Machine Learning', level: 'Intermediate', definition: 'When production data distribution changes over time.', learnMore: 'Drift detection triggers retraining and model governance workflows.' },
  { term: 'Precision and Recall', category: 'Machine Learning', level: 'Intermediate', definition: 'Metrics balancing false positives and false negatives.', learnMore: 'Precision measures correctness of positive predictions; recall measures coverage.' },

  { term: 'Accuracy', category: 'Machine Learning', level: 'Beginner', definition: 'Share of predictions that are correct.', learnMore: 'Accuracy can be misleading on imbalanced datasets.' },
  { term: 'F1 Score', category: 'Machine Learning', level: 'Intermediate', definition: 'Harmonic mean of precision and recall.', learnMore: 'F1 is useful when both false positives and false negatives matter.' },
  { term: 'Confusion Matrix', category: 'Machine Learning', level: 'Beginner', definition: 'Table of true/false positives and negatives.', learnMore: 'It helps diagnose where a classifier makes mistakes.' },
  { term: 'ROC Curve', category: 'Machine Learning', level: 'Intermediate', definition: 'Plot of true positive rate versus false positive rate.', learnMore: 'ROC illustrates threshold tradeoffs for binary classifiers.' },
  { term: 'AUC', category: 'Machine Learning', level: 'Intermediate', definition: 'Area under the ROC curve.', learnMore: 'Higher AUC means better ranking ability across thresholds.' },
  { term: 'Cross-Validation', category: 'Machine Learning', level: 'Intermediate', definition: 'Repeated train/validate splits for robust evaluation.', learnMore: 'K-fold cross-validation reduces variance in performance estimates.' },
  { term: 'Regularization', category: 'Machine Learning', level: 'Intermediate', definition: 'Penalty added to reduce overfitting.', learnMore: 'L1 and L2 regularization constrain model complexity.' },
  { term: 'L1 Regularization', category: 'Machine Learning', level: 'Advanced', definition: 'Penalty proportional to absolute weight values.', learnMore: 'L1 can drive some weights to zero, enabling sparse models.' },
  { term: 'L2 Regularization', category: 'Machine Learning', level: 'Advanced', definition: 'Penalty proportional to squared weight values.', learnMore: 'L2 discourages large weights and often improves generalization.' },
  { term: 'Dropout', category: 'Machine Learning', level: 'Intermediate', definition: 'Temporarily disabling random neurons during training.', learnMore: 'Dropout reduces co-adaptation and helps prevent overfitting.' },
  { term: 'Batch Normalization', category: 'Machine Learning', level: 'Advanced', definition: 'Normalizing activations within mini-batches.', learnMore: 'It stabilizes training and can speed convergence.' },
  { term: 'Learning Rate', category: 'Machine Learning', level: 'Beginner', definition: 'Step size used when updating model parameters.', learnMore: 'Too high can diverge; too low can slow training.' },
  { term: 'Learning Rate Scheduler', category: 'Machine Learning', level: 'Intermediate', definition: 'Rule that changes learning rate over training.', learnMore: 'Schedules like cosine decay often improve final performance.' },
  { term: 'Epoch', category: 'Machine Learning', level: 'Beginner', definition: 'One full pass through the training dataset.', learnMore: 'Models typically train across multiple epochs.' },
  { term: 'Batch Size', category: 'Machine Learning', level: 'Beginner', definition: 'Number of examples processed per optimization step.', learnMore: 'Batch size impacts memory use, speed, and generalization.' },
  { term: 'Loss Function', category: 'Machine Learning', level: 'Beginner', definition: 'Function measuring prediction error.', learnMore: 'Training seeks parameters that minimize loss.' },
  { term: 'Mean Squared Error (MSE)', category: 'Machine Learning', level: 'Intermediate', definition: 'Average of squared prediction errors.', learnMore: 'Common regression loss that penalizes large errors strongly.' },
  { term: 'Cross-Entropy Loss', category: 'Machine Learning', level: 'Intermediate', definition: 'Classification loss based on predicted probabilities.', learnMore: 'Lower cross-entropy means better probability calibration for labels.' },
  { term: 'Feature Engineering', category: 'Machine Learning', level: 'Intermediate', definition: 'Creating informative input variables from raw data.', learnMore: 'Good features can dramatically improve model performance.' },
  { term: 'Feature Selection', category: 'Machine Learning', level: 'Intermediate', definition: 'Choosing the most useful input features.', learnMore: 'It can improve interpretability and reduce overfitting.' },
  { term: 'Label', category: 'Machine Learning', level: 'Beginner', definition: 'Ground-truth output a model is trained to predict.', learnMore: 'Label quality directly impacts supervised learning results.' },
  { term: 'Class Imbalance', category: 'Machine Learning', level: 'Intermediate', definition: 'When some classes have many more examples than others.', learnMore: 'Resampling and class weights are common mitigation methods.' },
  { term: 'Ensemble Learning', category: 'Machine Learning', level: 'Intermediate', definition: 'Combining multiple models to improve predictions.', learnMore: 'Ensembles often reduce variance and improve robustness.' },
  { term: 'Bagging', category: 'Machine Learning', level: 'Intermediate', definition: 'Training multiple models on bootstrap samples.', learnMore: 'Random forests are a classic bagging approach.' },
  { term: 'Boosting', category: 'Machine Learning', level: 'Advanced', definition: 'Sequentially training weak learners to correct prior errors.', learnMore: 'Gradient boosting can deliver strong tabular-data performance.' },
  { term: 'Random Forest', category: 'Machine Learning', level: 'Intermediate', definition: 'An ensemble of decision trees trained with bagging.', learnMore: 'It is robust, interpretable, and effective on many tabular tasks.' },
  { term: 'Decision Tree', category: 'Machine Learning', level: 'Beginner', definition: 'Model that splits data by feature thresholds.', learnMore: 'Trees are easy to interpret but can overfit without pruning.' },
  { term: 'Support Vector Machine (SVM)', category: 'Machine Learning', level: 'Advanced', definition: 'Model that finds a margin-maximizing decision boundary.', learnMore: 'Kernel tricks allow nonlinear separation in transformed spaces.' },
  { term: 'K-Nearest Neighbors (KNN)', category: 'Machine Learning', level: 'Beginner', definition: 'Predicting based on labels of nearby data points.', learnMore: 'KNN is simple but can be slow on large datasets.' },
  { term: 'Naive Bayes', category: 'Machine Learning', level: 'Beginner', definition: 'Probabilistic classifier assuming conditional feature independence.', learnMore: 'Despite simplifications, Naive Bayes works well for some text tasks.' },
  { term: 'Logistic Regression', category: 'Machine Learning', level: 'Beginner', definition: 'Linear classifier that outputs class probabilities.', learnMore: 'It is a strong baseline with high interpretability.' },
  { term: 'Linear Regression', category: 'Machine Learning', level: 'Beginner', definition: 'Model fitting a linear relationship to predict continuous values.', learnMore: 'Linear regression is foundational for predictive analytics.' },
  { term: 'Dimensionality Reduction', category: 'Machine Learning', level: 'Intermediate', definition: 'Reducing feature space while retaining key information.', learnMore: 'Useful for visualization, denoising, and faster training.' },
  { term: 'PCA', category: 'Machine Learning', level: 'Intermediate', definition: 'Principal Component Analysis for linear dimensionality reduction.', learnMore: 'PCA projects data onto orthogonal directions of maximum variance.' },
  { term: 't-SNE', category: 'Machine Learning', level: 'Advanced', definition: 'Technique for visualizing high-dimensional data in low dimensions.', learnMore: 't-SNE preserves local neighborhoods for exploratory analysis.' },
  { term: 'UMAP', category: 'Machine Learning', level: 'Advanced', definition: 'Manifold-learning method for dimensionality reduction.', learnMore: 'UMAP is faster than t-SNE and preserves structure well.' },
  { term: 'Clustering', category: 'Machine Learning', level: 'Beginner', definition: 'Grouping unlabeled data points by similarity.', learnMore: 'Clustering reveals structure in data without target labels.' },
  { term: 'K-Means', category: 'Machine Learning', level: 'Beginner', definition: 'Clustering algorithm that partitions data into K groups.', learnMore: 'It iteratively updates centroids and assignments.' },
  { term: 'DBSCAN', category: 'Machine Learning', level: 'Advanced', definition: 'Density-based clustering algorithm.', learnMore: 'DBSCAN can find arbitrary-shaped clusters and identify noise.' },
  { term: 'Anomaly Detection', category: 'Machine Learning', level: 'Intermediate', definition: 'Finding unusual observations in data.', learnMore: 'Used in fraud detection, monitoring, and cybersecurity.' },
  { term: 'Calibration', category: 'Machine Learning', level: 'Advanced', definition: 'Alignment between predicted probabilities and observed outcomes.', learnMore: 'Calibrated probabilities improve decision-making under uncertainty.' },
  { term: 'Early Stopping', category: 'Machine Learning', level: 'Intermediate', definition: 'Stopping training when validation performance stops improving.', learnMore: 'This prevents overfitting and saves compute.' },
  { term: 'Data Augmentation', category: 'Machine Learning', level: 'Intermediate', definition: 'Creating modified examples to expand training data.', learnMore: 'Augmentation improves robustness, especially in vision and audio.' },
  { term: 'Active Learning', category: 'Machine Learning', level: 'Advanced', definition: 'Selecting the most informative samples for labeling.', learnMore: 'It reduces annotation cost by querying uncertain examples.' },
  { term: 'Semi-Supervised Learning', category: 'Machine Learning', level: 'Advanced', definition: 'Training with a mix of labeled and unlabeled data.', learnMore: 'Useful when labels are expensive but unlabeled data is abundant.' },
  { term: 'Self-Supervised Learning', category: 'Machine Learning', level: 'Advanced', definition: 'Learning representations from unlabeled data via pretext tasks.', learnMore: 'Modern foundation models heavily use self-supervised objectives.' },
  { term: 'Zero-Shot Learning', category: 'Machine Learning', level: 'Advanced', definition: 'Generalizing to tasks with no task-specific training examples.', learnMore: 'Large pretrained models enable strong zero-shot behavior.' },
  { term: 'Few-Shot Learning', category: 'Machine Learning', level: 'Advanced', definition: 'Learning effectively from very few examples.', learnMore: 'Prompting and metric-learning methods are common approaches.' },
  { term: 'Multimodal AI', category: 'Generative AI', level: 'Intermediate', definition: 'Models that process multiple data types, such as text and images.', learnMore: 'Multimodal systems enable richer reasoning and interfaces.' },
  { term: 'Inference', category: 'Machine Learning', level: 'Beginner', definition: 'Using a trained model to make predictions.', learnMore: 'Inference latency and throughput are key production metrics.' },
  { term: 'Latency', category: 'Machine Learning', level: 'Beginner', definition: 'Time it takes a model to return a prediction.', learnMore: 'Low latency is essential for interactive AI applications.' },
  { term: 'Throughput', category: 'Machine Learning', level: 'Beginner', definition: 'Number of predictions processed per time unit.', learnMore: 'High throughput matters for batch and high-traffic workloads.' },
  { term: 'Quantization', category: 'Machine Learning', level: 'Advanced', definition: 'Reducing numerical precision of model weights/activations.', learnMore: 'Quantization cuts memory use and can speed inference.' },
  { term: 'Pruning', category: 'Machine Learning', level: 'Advanced', definition: 'Removing less important model parameters.', learnMore: 'Pruning can shrink models with limited quality loss.' },
  { term: 'Knowledge Distillation', category: 'Machine Learning', level: 'Advanced', definition: 'Training a smaller student model from a larger teacher model.', learnMore: 'Distillation improves efficiency while retaining much of the teacher performance.' },
  { term: 'ONNX', category: 'Machine Learning', level: 'Intermediate', definition: 'Open format for interoperable model representation.', learnMore: 'ONNX helps deploy models across runtimes and hardware.' },
  { term: 'MLOps', category: 'Machine Learning', level: 'Intermediate', definition: 'Practices for deploying and operating ML systems reliably.', learnMore: 'MLOps includes CI/CD, monitoring, retraining, and governance.' },
  { term: 'Model Monitoring', category: 'Machine Learning', level: 'Intermediate', definition: 'Tracking model quality and behavior in production.', learnMore: 'Monitoring catches drift, outages, and quality regressions early.' },
  { term: 'Model Versioning', category: 'Machine Learning', level: 'Intermediate', definition: 'Tracking model artifacts and changes over time.', learnMore: 'Versioning enables rollback, auditing, and reproducibility.' },
  { term: 'Experiment Tracking', category: 'Machine Learning', level: 'Intermediate', definition: 'Recording runs, hyperparameters, and metrics.', learnMore: 'Tracking helps teams compare experiments and reproduce results.' },
  { term: 'Feature Store', category: 'Machine Learning', level: 'Advanced', definition: 'Central system for managing and serving ML features.', learnMore: 'Feature stores ensure train/serve consistency and reuse.' },
  { term: 'Data Labeling', category: 'Machine Learning', level: 'Beginner', definition: 'Annotating data with ground truth targets.', learnMore: 'Label quality control is crucial for reliable supervised models.' },
  { term: 'Ground Truth', category: 'Machine Learning', level: 'Beginner', definition: 'Reference-correct data used for training or evaluation.', learnMore: 'Noisy ground truth can cap model performance.' },
  { term: 'Covariate Shift', category: 'Machine Learning', level: 'Advanced', definition: 'Change in input distribution while label relationship remains stable.', learnMore: 'Importance weighting can partially correct covariate shift.' },
  { term: 'Concept Drift', category: 'Machine Learning', level: 'Advanced', definition: 'Change in relationship between features and target over time.', learnMore: 'Concept drift often requires retraining with recent data.' },
  { term: 'Cold Start', category: 'Machine Learning', level: 'Intermediate', definition: 'Lack of user/item data when launching recommendation systems.', learnMore: 'Hybrid methods and metadata features help mitigate cold start.' },
  { term: 'Recommender System', category: 'Machine Learning', level: 'Beginner', definition: 'System suggesting items users are likely to prefer.', learnMore: 'Recommendations can use collaborative, content-based, or hybrid methods.' },
  { term: 'Collaborative Filtering', category: 'Machine Learning', level: 'Intermediate', definition: 'Recommendation based on user-item interaction patterns.', learnMore: 'It leverages behavior similarity across users and items.' },
  { term: 'Content-Based Filtering', category: 'Machine Learning', level: 'Intermediate', definition: 'Recommendation using item attributes and user profiles.', learnMore: 'It works well when interaction history is sparse.' },

  { term: 'Language Modeling', category: 'NLP', level: 'Beginner', definition: 'Predicting token sequences in natural language.', learnMore: 'Language modeling is the core objective behind many modern LLMs.' },
  { term: 'Perplexity', category: 'NLP', level: 'Intermediate', definition: 'Metric of how well a language model predicts text.', learnMore: 'Lower perplexity typically indicates better next-token modeling.' },
  { term: 'Part-of-Speech Tagging', category: 'NLP', level: 'Beginner', definition: 'Labeling words by grammatical role.', learnMore: 'POS tagging supports parsing and information extraction pipelines.' },
  { term: 'Dependency Parsing', category: 'NLP', level: 'Advanced', definition: 'Analyzing grammatical relationships between words.', learnMore: 'Dependency trees represent sentence structure and head-dependent links.' },
  { term: 'Stemming', category: 'NLP', level: 'Beginner', definition: 'Reducing words to crude root forms.', learnMore: 'Stemming is fast but less precise than lemmatization.' },
  { term: 'Lemmatization', category: 'NLP', level: 'Intermediate', definition: 'Converting words to dictionary base forms.', learnMore: 'Lemmatization uses linguistic rules for cleaner normalization.' },
  { term: 'Stop Words', category: 'NLP', level: 'Beginner', definition: 'Very common words often removed in preprocessing.', learnMore: 'Examples include “the,” “is,” and “and,” depending on task.' },
  { term: 'N-gram', category: 'NLP', level: 'Beginner', definition: 'A contiguous sequence of n tokens.', learnMore: 'N-grams capture local context for classic NLP models.' },
  { term: 'Word2Vec', category: 'NLP', level: 'Intermediate', definition: 'Method for learning dense word embeddings.', learnMore: 'Word2Vec uses CBOW or skip-gram objectives.' },
  { term: 'GloVe', category: 'NLP', level: 'Intermediate', definition: 'Word embedding method based on global co-occurrence statistics.', learnMore: 'GloVe balances local context and global corpus structure.' },
  { term: 'BERT', category: 'NLP', level: 'Advanced', definition: 'Bidirectional transformer pretrained with masked language modeling.', learnMore: 'BERT improved many NLP benchmarks via contextual representations.' },
  { term: 'GPT', category: 'Generative AI', level: 'Intermediate', definition: 'Decoder-only transformer family for text generation.', learnMore: 'GPT models are pretrained autoregressively and adapted via prompting or tuning.' },
  { term: 'T5', category: 'NLP', level: 'Advanced', definition: 'Text-to-text transformer framing all tasks as generation.', learnMore: 'T5 unifies translation, summarization, and classification formats.' },
  { term: 'Seq2Seq', category: 'NLP', level: 'Intermediate', definition: 'Model mapping one sequence to another.', learnMore: 'Used for translation, summarization, and dialogue generation.' },
  { term: 'Beam Search', category: 'NLP', level: 'Advanced', definition: 'Decoding algorithm keeping top candidate sequences.', learnMore: 'Beam width controls quality-speed tradeoffs in generation.' },
  { term: 'Top-k Sampling', category: 'Generative AI', level: 'Intermediate', definition: 'Sampling next token from the top-k probable tokens.', learnMore: 'Limits unlikely outputs while preserving some diversity.' },
  { term: 'Top-p Sampling', category: 'Generative AI', level: 'Intermediate', definition: 'Sampling from smallest token set whose cumulative probability exceeds p.', learnMore: 'Also called nucleus sampling; it adapts candidate pool size dynamically.' },
  { term: 'Temperature', category: 'Generative AI', level: 'Beginner', definition: 'Parameter controlling randomness in generation.', learnMore: 'Lower temperature yields more deterministic outputs.' },
  { term: 'Context Window', category: 'Generative AI', level: 'Intermediate', definition: 'Maximum amount of input tokens a model can attend to.', learnMore: 'Longer context supports larger documents and conversation memory.' },
  { term: 'Prompt Injection', category: 'Ethics & Safety', level: 'Advanced', definition: 'Malicious instructions hidden in input to manipulate model behavior.', learnMore: 'Mitigations include isolation, filtering, and policy-aware tool use.' },
  { term: 'Jailbreak', category: 'Ethics & Safety', level: 'Advanced', definition: 'Attempt to bypass model safety constraints.', learnMore: 'Defense requires robust policies, monitoring, and adversarial testing.' },
  { term: 'PII Detection', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Identifying personally identifiable information in text/data.', learnMore: 'PII handling is key for privacy, compliance, and safe deployments.' },
  { term: 'Redaction', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Removing sensitive information from data or outputs.', learnMore: 'Automated redaction helps protect personal and confidential data.' },
  { term: 'Toxicity Detection', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Detecting harmful, abusive, or unsafe language.', learnMore: 'Often used in moderation and safety pipelines.' },
  { term: 'Machine Translation', category: 'NLP', level: 'Beginner', definition: 'Automatically translating text between languages.', learnMore: 'Modern systems use transformers and large multilingual datasets.' },
  { term: 'Summarization', category: 'NLP', level: 'Beginner', definition: 'Generating concise versions of longer content.', learnMore: 'Can be extractive or abstractive depending on method.' },
  { term: 'Question Answering', category: 'NLP', level: 'Beginner', definition: 'Answering questions from text, knowledge, or context.', learnMore: 'QA systems power assistants, search, and enterprise support tools.' },
  { term: 'Text Classification', category: 'NLP', level: 'Beginner', definition: 'Assigning category labels to text inputs.', learnMore: 'Common tasks include spam detection, intent routing, and topic tagging.' },
  { term: 'Topic Modeling', category: 'NLP', level: 'Intermediate', definition: 'Discovering latent themes in document collections.', learnMore: 'LDA is a classic probabilistic topic modeling approach.' },
  { term: 'Latent Dirichlet Allocation (LDA)', category: 'NLP', level: 'Advanced', definition: 'Probabilistic model representing documents as topic mixtures.', learnMore: 'LDA infers hidden topics and associated word distributions.' },
  { term: 'ROUGE', category: 'NLP', level: 'Intermediate', definition: 'Metric family for summarization quality via overlap with references.', learnMore: 'ROUGE emphasizes recall of overlapping units.' },
  { term: 'METEOR', category: 'NLP', level: 'Advanced', definition: 'Translation metric using stemming, synonyms, and alignment.', learnMore: 'METEOR can correlate better with human judgment than raw n-gram overlap.' },
  { term: 'Word Error Rate (WER)', category: 'NLP', level: 'Intermediate', definition: 'ASR metric based on insertions, deletions, and substitutions.', learnMore: 'Lower WER indicates better speech recognition accuracy.' },
  { term: 'Speech Recognition', category: 'NLP', level: 'Beginner', definition: 'Converting spoken audio into text.', learnMore: 'Also called automatic speech recognition (ASR).' },
  { term: 'Text-to-Speech (TTS)', category: 'NLP', level: 'Beginner', definition: 'Generating natural-sounding speech from text.', learnMore: 'Modern neural TTS improves prosody and voice quality.' },
  { term: 'Speaker Diarization', category: 'NLP', level: 'Advanced', definition: 'Determining who spoke when in audio streams.', learnMore: 'Useful in meetings, call analytics, and transcription workflows.' },

  { term: 'Image Classification', category: 'Computer Vision', level: 'Beginner', definition: 'Assigning a label to an entire image.', learnMore: 'It is one of the foundational vision tasks.' },
  { term: 'Semantic Segmentation', category: 'Computer Vision', level: 'Advanced', definition: 'Pixel-wise classification by semantic category.', learnMore: 'Unlike instance segmentation, it does not separate object instances.' },
  { term: 'Instance Segmentation', category: 'Computer Vision', level: 'Advanced', definition: 'Pixel-wise segmentation that distinguishes individual instances.', learnMore: 'Mask R-CNN is a well-known instance segmentation model.' },
  { term: 'Bounding Box', category: 'Computer Vision', level: 'Beginner', definition: 'Rectangle indicating object location in an image.', learnMore: 'Bounding boxes are common labels for detection datasets.' },
  { term: 'Intersection over Union (IoU)', category: 'Computer Vision', level: 'Intermediate', definition: 'Overlap metric between predicted and true regions.', learnMore: 'Higher IoU indicates better localization quality.' },
  { term: 'mAP', category: 'Computer Vision', level: 'Advanced', definition: 'Mean Average Precision used to evaluate detection models.', learnMore: 'mAP summarizes precision-recall behavior across classes and thresholds.' },
  { term: 'Optical Flow', category: 'Computer Vision', level: 'Advanced', definition: 'Estimating motion of pixels between video frames.', learnMore: 'Optical flow supports tracking and action understanding.' },
  { term: 'Pose Estimation', category: 'Computer Vision', level: 'Intermediate', definition: 'Predicting body or object keypoint locations.', learnMore: 'Used in sports analytics, AR, and human-computer interaction.' },
  { term: 'Facial Recognition', category: 'Computer Vision', level: 'Intermediate', definition: 'Identifying or verifying people from facial images.', learnMore: 'Requires careful governance due to privacy and fairness risks.' },
  { term: 'OCR', category: 'Computer Vision', level: 'Beginner', definition: 'Optical Character Recognition, converting text in images to machine text.', learnMore: 'OCR enables document digitization and data extraction.' },
  { term: 'Vision Transformer (ViT)', category: 'Computer Vision', level: 'Advanced', definition: 'Transformer architecture applied to image patches.', learnMore: 'ViTs are strong alternatives to CNNs at scale.' },
  { term: 'Data Annotation', category: 'Computer Vision', level: 'Beginner', definition: 'Labeling images with classes, boxes, masks, or keypoints.', learnMore: 'Annotation quality is central to vision model performance.' },
  { term: 'YOLO', category: 'Computer Vision', level: 'Intermediate', definition: 'Real-time object detection model family.', learnMore: 'YOLO predicts boxes and classes in a single forward pass.' },
  { term: 'R-CNN', category: 'Computer Vision', level: 'Advanced', definition: 'Region-based convolutional object detection approach.', learnMore: 'Faster R-CNN improved speed with region proposal networks.' },
  { term: 'U-Net', category: 'Computer Vision', level: 'Advanced', definition: 'Encoder-decoder network architecture for segmentation.', learnMore: 'U-Net is popular in biomedical image segmentation.' },
  { term: 'Autoencoder', category: 'Machine Learning', level: 'Advanced', definition: 'Neural network trained to reconstruct inputs via compressed representations.', learnMore: 'Autoencoders are used for denoising, compression, and anomaly detection.' },
  { term: 'Variational Autoencoder (VAE)', category: 'Generative AI', level: 'Advanced', definition: 'Probabilistic autoencoder for generative modeling.', learnMore: 'VAEs learn latent spaces that can be sampled to generate new data.' },
  { term: 'Generative Adversarial Network (GAN)', category: 'Generative AI', level: 'Advanced', definition: 'Generative framework with competing generator and discriminator networks.', learnMore: 'GANs can create realistic images but may be hard to train stably.' },
  { term: 'Diffusion Model', category: 'Generative AI', level: 'Advanced', definition: 'Generative model that denoises random noise into data samples.', learnMore: 'Diffusion models power many modern text-to-image systems.' },
  { term: 'Latent Space', category: 'Generative AI', level: 'Intermediate', definition: 'Compressed representation where high-level data factors are encoded.', learnMore: 'Operations in latent space can control generation properties.' },
  { term: 'Sampling', category: 'Generative AI', level: 'Beginner', definition: 'Process of generating outputs from a model distribution.', learnMore: 'Sampling strategy controls creativity and determinism.' },
  { term: 'Classifier-Free Guidance', category: 'Generative AI', level: 'Advanced', definition: 'Technique guiding diffusion outputs toward prompt conditions.', learnMore: 'Higher guidance can improve prompt fidelity but reduce diversity.' },
  { term: 'Text-to-Image', category: 'Generative AI', level: 'Beginner', definition: 'Generating images from natural language prompts.', learnMore: 'Systems combine language understanding with visual generation models.' },
  { term: 'Image-to-Image', category: 'Generative AI', level: 'Intermediate', definition: 'Transforming an input image based on instructions or style.', learnMore: 'Used for editing, style transfer, and domain translation.' },
  { term: 'Inpainting', category: 'Generative AI', level: 'Intermediate', definition: 'Filling missing or masked image regions.', learnMore: 'Inpainting enables object removal and content-aware editing.' },
  { term: 'Outpainting', category: 'Generative AI', level: 'Intermediate', definition: 'Extending image content beyond original boundaries.', learnMore: 'Outpainting generates coherent new context around an image.' },
  { term: 'Style Transfer', category: 'Generative AI', level: 'Intermediate', definition: 'Applying artistic style of one image to another.', learnMore: 'Neural style transfer separates content and style representations.' },
  { term: 'LoRA', category: 'Generative AI', level: 'Advanced', definition: 'Low-Rank Adaptation method for parameter-efficient fine-tuning.', learnMore: 'LoRA adds trainable low-rank matrices while freezing base weights.' },
  { term: 'PEFT', category: 'Generative AI', level: 'Advanced', definition: 'Parameter-Efficient Fine-Tuning techniques for large models.', learnMore: 'PEFT methods reduce memory and compute costs of adaptation.' },
  { term: 'RLHF', category: 'Generative AI', level: 'Advanced', definition: 'Reinforcement Learning from Human Feedback.', learnMore: 'RLHF aligns model behavior with human preferences.' },
  { term: 'DPO', category: 'Generative AI', level: 'Advanced', definition: 'Direct Preference Optimization using preference pairs.', learnMore: 'DPO optimizes preference alignment without full RL pipelines.' },
  { term: 'System Prompt', category: 'Generative AI', level: 'Beginner', definition: 'Top-level instruction guiding assistant behavior.', learnMore: 'System prompts establish policy, tone, and task constraints.' },
  { term: 'Function Calling', category: 'Generative AI', level: 'Intermediate', definition: 'Structured mechanism for models to request tool execution.', learnMore: 'It improves reliability for actions like search or database queries.' },
  { term: 'Tool Use', category: 'Generative AI', level: 'Intermediate', definition: 'Allowing models to call external APIs or software tools.', learnMore: 'Tool use extends capabilities beyond model internal knowledge.' },
  { term: 'Agentic AI', category: 'Generative AI', level: 'Advanced', definition: 'AI systems that plan and execute multi-step actions autonomously.', learnMore: 'Agentic flows combine reasoning, memory, and tool execution loops.' },
  { term: 'Chain-of-Thought', category: 'Generative AI', level: 'Intermediate', definition: 'Reasoning process broken into intermediate steps.', learnMore: 'Prompting for stepwise reasoning can improve complex task performance.' },
  { term: 'ReAct', category: 'Generative AI', level: 'Advanced', definition: 'Framework combining reasoning and actions iteratively.', learnMore: 'ReAct alternates thinking traces with tool calls.' },
  { term: 'Evaluation Harness', category: 'Generative AI', level: 'Intermediate', definition: 'Suite for systematic model benchmark testing.', learnMore: 'Harnesses automate repeatable quality checks across tasks.' },
  { term: 'Benchmark', category: 'Machine Learning', level: 'Beginner', definition: 'Standardized test set used to compare models.', learnMore: 'Benchmarks help track progress but can be overfit over time.' },
  { term: 'Leaderboard', category: 'Machine Learning', level: 'Beginner', definition: 'Ranking of models by benchmark performance.', learnMore: 'Leaderboards encourage progress but do not capture all real-world needs.' },

  { term: 'AI Governance', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Policies and controls for responsible AI lifecycle management.', learnMore: 'Governance includes risk management, accountability, and oversight.' },
  { term: 'Model Card', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Structured documentation of model behavior and limitations.', learnMore: 'Model cards improve transparency for users and auditors.' },
  { term: 'Datasheet for Datasets', category: 'Ethics & Safety', level: 'Advanced', definition: 'Documentation template describing dataset motivation and collection.', learnMore: 'Datasheets support better dataset transparency and responsible reuse.' },
  { term: 'Privacy-Preserving ML', category: 'Ethics & Safety', level: 'Advanced', definition: 'ML methods that protect sensitive information.', learnMore: 'Includes federated learning, differential privacy, and secure computation.' },
  { term: 'Differential Privacy', category: 'Ethics & Safety', level: 'Advanced', definition: 'Formal privacy guarantee limiting individual data influence.', learnMore: 'Noise injection enables aggregate learning with reduced reidentification risk.' },
  { term: 'Federated Learning', category: 'Ethics & Safety', level: 'Advanced', definition: 'Training across devices without centralizing raw data.', learnMore: 'Only model updates are shared, helping privacy and data locality.' },
  { term: 'Secure Aggregation', category: 'Ethics & Safety', level: 'Advanced', definition: 'Cryptographic protocol to aggregate client updates privately.', learnMore: 'Used in federated learning to hide individual contributions.' },
  { term: 'Model Inversion Attack', category: 'Ethics & Safety', level: 'Advanced', definition: 'Attack reconstructing training data signals from model outputs.', learnMore: 'Mitigated with privacy-aware training and output controls.' },
  { term: 'Membership Inference Attack', category: 'Ethics & Safety', level: 'Advanced', definition: 'Attack predicting whether a sample was in training data.', learnMore: 'Overfitting often increases vulnerability to membership inference.' },
  { term: 'Data Poisoning', category: 'Ethics & Safety', level: 'Advanced', definition: 'Corrupting training data to degrade or manipulate model behavior.', learnMore: 'Robust curation and anomaly checks reduce poisoning risk.' },
  { term: 'Backdoor Attack', category: 'Ethics & Safety', level: 'Advanced', definition: 'Embedding hidden triggers that cause malicious model behavior.', learnMore: 'Backdoors can remain dormant until specific trigger patterns appear.' },
  { term: 'Robustness', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Model reliability under noise, shifts, or adversarial conditions.', learnMore: 'Robustness testing should be part of pre-deployment evaluation.' },
  { term: 'Safety Filter', category: 'Ethics & Safety', level: 'Intermediate', definition: 'System that blocks unsafe prompts or outputs.', learnMore: 'Filters complement model-level alignment and policy controls.' },
  { term: 'Human-in-the-Loop', category: 'Ethics & Safety', level: 'Beginner', definition: 'Including human oversight in AI decision workflows.', learnMore: 'Human review improves safety for high-stakes applications.' },
  { term: 'Responsible AI', category: 'Ethics & Safety', level: 'Beginner', definition: 'Developing AI with fairness, safety, and accountability principles.', learnMore: 'Responsible AI spans design, deployment, and post-launch monitoring.' },
  { term: 'Transparency', category: 'Ethics & Safety', level: 'Beginner', definition: 'Clear communication about how AI systems work and are used.', learnMore: 'Transparency helps users understand limitations and risks.' },
  { term: 'Accountability', category: 'Ethics & Safety', level: 'Beginner', definition: 'Assigning responsibility for AI outcomes and decisions.', learnMore: 'Clear ownership ensures issues are addressed quickly and ethically.' },
  { term: 'Interpretability', category: 'Ethics & Safety', level: 'Intermediate', definition: 'How understandable a model’s internal logic is to humans.', learnMore: 'Interpretable models are often preferred in regulated domains.' },
  { term: 'SHAP', category: 'Ethics & Safety', level: 'Advanced', definition: 'Method attributing prediction impact to input features.', learnMore: 'SHAP values provide local and global explanation perspectives.' },
  { term: 'LIME', category: 'Ethics & Safety', level: 'Advanced', definition: 'Local surrogate method for explaining individual predictions.', learnMore: 'LIME approximates complex models around a specific example.' },
  { term: 'Risk Assessment', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Evaluating potential harms before and after deployment.', learnMore: 'Risk assessment informs safeguards, controls, and escalation procedures.' },
  { term: 'Compliance', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Adhering to laws, standards, and internal AI policies.', learnMore: 'Compliance programs require auditing, documentation, and governance evidence.' },
  { term: 'Audit Trail', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Record of model versions, data changes, and decisions.', learnMore: 'Audit trails support accountability and incident investigations.' },
  { term: 'Red Teaming', category: 'Ethics & Safety', level: 'Advanced', definition: 'Adversarial testing to find system weaknesses.', learnMore: 'Red teaming improves safety by exposing failure modes early.' },
  { term: 'Guardrails', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Constraints and checks that keep model behavior within policy.', learnMore: 'Guardrails may include prompt checks, output validation, and tool restrictions.' },
  { term: 'Content Moderation', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Detecting and managing harmful or policy-violating content.', learnMore: 'Moderation combines classifiers, rules, and human review.' },
  { term: 'Synthetic Data', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Artificially generated data used for training/testing.', learnMore: 'Synthetic data can improve privacy and cover rare scenarios.' },
  { term: 'Watermarking', category: 'Ethics & Safety', level: 'Advanced', definition: 'Embedding signals to identify AI-generated content.', learnMore: 'Watermarking can support provenance and misuse detection.' },
  { term: 'Provenance', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Traceability of data or content origin and transformations.', learnMore: 'Provenance helps establish trust and accountability.' },
  { term: 'AI Incident', category: 'Ethics & Safety', level: 'Beginner', definition: 'Event where AI causes or contributes to harm or policy breach.', learnMore: 'Incident response plans should include rollback and communication steps.' },
  { term: 'Fallback Policy', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Safe alternative behavior when model confidence is low.', learnMore: 'Fallbacks often route to humans or conservative responses.' },
  { term: 'Confidence Score', category: 'Machine Learning', level: 'Beginner', definition: 'Model-estimated certainty for a prediction.', learnMore: 'Confidence is not always calibrated and should be validated.' },
  { term: 'Abstention', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Choosing not to answer when uncertainty or risk is high.', learnMore: 'Abstention can reduce harmful high-confidence errors.' },
  { term: 'Instruction Tuning', category: 'Generative AI', level: 'Advanced', definition: 'Fine-tuning models on instruction-response pairs.', learnMore: 'Instruction tuning improves task following and assistant usefulness.' },
  { term: 'Supervised Fine-Tuning (SFT)', category: 'Generative AI', level: 'Advanced', definition: 'Fine-tuning with labeled input-output examples.', learnMore: 'SFT typically precedes preference alignment stages.' },
  { term: 'Contextual Bandit', category: 'Machine Learning', level: 'Advanced', definition: 'Decision framework balancing exploration and exploitation with context.', learnMore: 'Used for online personalization and recommendation optimization.' },
  { term: 'Markov Decision Process (MDP)', category: 'Machine Learning', level: 'Advanced', definition: 'Formal model of states, actions, transitions, and rewards.', learnMore: 'MDPs are the mathematical foundation of many RL algorithms.' },
  { term: 'Q-Learning', category: 'Machine Learning', level: 'Advanced', definition: 'RL algorithm learning action values without a model of environment dynamics.', learnMore: 'Q-learning updates value estimates via temporal-difference learning.' },
  { term: 'Policy Gradient', category: 'Machine Learning', level: 'Advanced', definition: 'RL method directly optimizing policy parameters.', learnMore: 'Policy gradients handle continuous actions naturally.' },
  { term: 'Actor-Critic', category: 'Machine Learning', level: 'Advanced', definition: 'RL architecture combining policy (actor) and value (critic) models.', learnMore: 'The critic reduces variance in policy updates.' },
  { term: 'Reward Function', category: 'Machine Learning', level: 'Intermediate', definition: 'Signal defining desired behavior in reinforcement learning.', learnMore: 'Poorly specified rewards can cause unintended behaviors.' },
  { term: 'Reward Hacking', category: 'Ethics & Safety', level: 'Advanced', definition: 'When an agent exploits reward loopholes instead of intended goals.', learnMore: 'Robust reward design and oversight are needed to prevent it.' },
  { term: 'Generalization', category: 'Machine Learning', level: 'Beginner', definition: 'Ability to perform well on unseen data.', learnMore: 'Generalization is the central objective of model training.' },
  { term: 'Out-of-Distribution (OOD)', category: 'Machine Learning', level: 'Advanced', definition: 'Inputs that differ from the training distribution.', learnMore: 'OOD detection is important for safe deployment.' },
  { term: 'Domain Adaptation', category: 'Machine Learning', level: 'Advanced', definition: 'Adapting models trained in one domain to another domain.', learnMore: 'It addresses distribution shifts between source and target data.' },
  { term: 'Domain Generalization', category: 'Machine Learning', level: 'Advanced', definition: 'Training models to generalize to unseen domains.', learnMore: 'Methods often rely on invariant feature learning.' },
  { term: 'Curriculum Learning', category: 'Machine Learning', level: 'Advanced', definition: 'Training from easier to harder examples over time.', learnMore: 'Curriculum schedules can improve convergence and stability.' },
  { term: 'Catastrophic Forgetting', category: 'Machine Learning', level: 'Advanced', definition: 'Loss of old knowledge when learning new tasks.', learnMore: 'Continual learning methods mitigate forgetting.' },
  { term: 'Continual Learning', category: 'Machine Learning', level: 'Advanced', definition: 'Learning from data streams without retraining from scratch.', learnMore: 'Continual learning targets adaptability with memory retention.' },
  { term: 'Retrieval System', category: 'Generative AI', level: 'Intermediate', definition: 'Component that fetches relevant documents for queries.', learnMore: 'Retrieval quality strongly affects RAG answer quality.' },
  { term: 'Vector Database', category: 'Generative AI', level: 'Intermediate', definition: 'Database optimized for nearest-neighbor search on embeddings.', learnMore: 'Vector DBs power semantic search and retrieval pipelines.' },
  { term: 'Cosine Similarity', category: 'Generative AI', level: 'Beginner', definition: 'Similarity measure based on angle between vectors.', learnMore: 'Commonly used for comparing embeddings.' },
  { term: 'Approximate Nearest Neighbor (ANN)', category: 'Generative AI', level: 'Advanced', definition: 'Fast search for near vectors with acceptable approximation error.', learnMore: 'ANN indexing scales vector search to millions or billions of items.' },
  { term: 'Chunking', category: 'Generative AI', level: 'Intermediate', definition: 'Splitting documents into smaller segments for retrieval.', learnMore: 'Good chunking improves recall and context relevance in RAG.' },
  { term: 'Re-ranking', category: 'Generative AI', level: 'Advanced', definition: 'Second-stage model that reorders retrieved candidates for relevance.', learnMore: 'Re-rankers improve precision before final answer generation.' },
  { term: 'Grounding', category: 'Generative AI', level: 'Intermediate', definition: 'Linking model outputs to verified external evidence.', learnMore: 'Grounded generation reduces hallucinations and improves trust.' },
  { term: 'Agent Memory', category: 'Generative AI', level: 'Intermediate', definition: 'Stored context that helps an agent maintain continuity across tasks.', learnMore: 'Memory may be short-term context or long-term external storage.' },
  { term: 'Toolformer', category: 'Generative AI', level: 'Advanced', definition: 'Model paradigm that learns when and how to call tools.', learnMore: 'Tool-aware models can improve factual accuracy and action reliability.' },
  { term: 'Prompt Template', category: 'Generative AI', level: 'Beginner', definition: 'Reusable prompt structure with placeholders.', learnMore: 'Templates standardize interactions and reduce prompt variability.' },
  { term: 'Hallucination Rate', category: 'Generative AI', level: 'Intermediate', definition: 'Frequency of fabricated or incorrect model statements.', learnMore: 'Tracking hallucination rate supports quality governance.' },
  { term: 'Safety Benchmark', category: 'Ethics & Safety', level: 'Advanced', definition: 'Evaluation suite focused on harmful behavior and policy adherence.', learnMore: 'Safety benchmarks test robustness against risky prompts and attacks.' },
  { term: 'Data Lineage', category: 'Ethics & Safety', level: 'Intermediate', definition: 'Trace of how data moves and transforms through systems.', learnMore: 'Lineage supports debugging, compliance, and trust.' },
  { term: 'Observability', category: 'Machine Learning', level: 'Intermediate', definition: 'Comprehensive visibility into model/system state and behavior.', learnMore: 'Observability combines logs, metrics, traces, and alerts.' },
  { term: 'Canary Deployment', category: 'Machine Learning', level: 'Intermediate', definition: 'Gradually rolling out a new model to a small traffic subset.', learnMore: 'Canaries reduce risk before full production rollout.' },
  { term: 'A/B Testing', category: 'Machine Learning', level: 'Intermediate', definition: 'Comparing two model versions with randomized user traffic.', learnMore: 'A/B tests measure real-world impact beyond offline metrics.' },
  { term: 'Shadow Deployment', category: 'Machine Learning', level: 'Advanced', definition: 'Running a new model in parallel without affecting users.', learnMore: 'Shadow mode enables safe production evaluation.' },
  { term: 'Rollback', category: 'Machine Learning', level: 'Beginner', definition: 'Reverting to a previous stable model version.', learnMore: 'Fast rollback is essential for incident response.' },
  { term: 'SLA', category: 'Machine Learning', level: 'Beginner', definition: 'Service Level Agreement defining reliability/performance targets.', learnMore: 'AI services may define SLA for latency, uptime, and quality.' },
  { term: 'SLI', category: 'Machine Learning', level: 'Intermediate', definition: 'Service Level Indicator measuring service performance.', learnMore: 'Examples include p95 latency and error rate.' },
  { term: 'SLO', category: 'Machine Learning', level: 'Intermediate', definition: 'Service Level Objective target for an indicator.', learnMore: 'SLOs guide alerting and operational priorities.' },
  { term: 'Prompt Caching', category: 'Generative AI', level: 'Intermediate', definition: 'Reusing previous prompt computations to reduce latency/cost.', learnMore: 'Caching benefits repetitive workloads and long contexts.' },
  { term: 'Token Budget', category: 'Generative AI', level: 'Beginner', definition: 'Maximum tokens allocated for prompt and response.', learnMore: 'Managing token budget controls cost and truncation risk.' },
  { term: 'Rate Limiting', category: 'Machine Learning', level: 'Beginner', definition: 'Controlling request frequency to protect system stability.', learnMore: 'Rate limits prevent abuse and maintain service quality.' },
  { term: 'Cold Path / Hot Path', category: 'Machine Learning', level: 'Intermediate', definition: 'Slow offline processing path versus fast online serving path.', learnMore: 'Separating paths helps optimize for both cost and latency.' },
  { term: 'Edge AI', category: 'Machine Learning', level: 'Intermediate', definition: 'Running AI models directly on local devices.', learnMore: 'Edge AI reduces latency and can improve privacy.' },
  { term: 'TinyML', category: 'Machine Learning', level: 'Advanced', definition: 'Machine learning on ultra-low-power microcontrollers.', learnMore: 'TinyML enables always-on intelligence in constrained hardware.' },
  { term: 'FP16', category: 'Machine Learning', level: 'Advanced', definition: '16-bit floating-point precision used for faster training/inference.', learnMore: 'Mixed precision can speed compute while preserving quality.' },
  { term: 'BF16', category: 'Machine Learning', level: 'Advanced', definition: 'bfloat16 format with wider exponent range than FP16.', learnMore: 'BF16 is popular for stable large-model training on modern accelerators.' },
  { term: 'GPU', category: 'Machine Learning', level: 'Beginner', definition: 'Parallel processor widely used for AI training and inference.', learnMore: 'GPUs accelerate matrix operations central to deep learning.' },
  { term: 'TPU', category: 'Machine Learning', level: 'Intermediate', definition: 'Specialized accelerator designed for tensor operations.', learnMore: 'TPUs are optimized for large-scale neural network workloads.' },
  { term: 'Distributed Training', category: 'Machine Learning', level: 'Advanced', definition: 'Training models across multiple devices or nodes.', learnMore: 'Data and model parallelism are common distributed strategies.' },
  { term: 'Data Parallelism', category: 'Machine Learning', level: 'Advanced', definition: 'Replicating a model across devices with different data shards.', learnMore: 'Gradients are synchronized to keep replicas aligned.' },
  { term: 'Model Parallelism', category: 'Machine Learning', level: 'Advanced', definition: 'Splitting one model across multiple devices.', learnMore: 'Useful when a single model does not fit on one device.' },
  { term: 'Checkpoint', category: 'Machine Learning', level: 'Beginner', definition: 'Saved snapshot of model weights during training.', learnMore: 'Checkpoints allow resume, rollback, and model selection.' },
  { term: 'Seed', category: 'Machine Learning', level: 'Beginner', definition: 'Initial value controlling pseudo-random operations.', learnMore: 'Fixing seeds improves reproducibility across runs.' },
  { term: 'Reproducibility', category: 'Machine Learning', level: 'Intermediate', definition: 'Ability to replicate results with same code and data.', learnMore: 'Essential for scientific rigor and dependable engineering.' },
  { term: 'Data Leakage', category: 'Machine Learning', level: 'Intermediate', definition: 'Unintended use of future/target information during training.', learnMore: 'Leakage inflates offline metrics and harms real-world performance.' },
  { term: 'Train/Test Split', category: 'Machine Learning', level: 'Beginner', definition: 'Partitioning data into training and evaluation subsets.', learnMore: 'Strict separation prevents optimistic bias in evaluation.' },
  { term: 'Validation Set', category: 'Machine Learning', level: 'Beginner', definition: 'Dataset used for model selection and hyperparameter tuning.', learnMore: 'Validation data should not be used for final unbiased reporting.' },
  { term: 'Holdout Set', category: 'Machine Learning', level: 'Intermediate', definition: 'Final untouched dataset for unbiased model evaluation.', learnMore: 'Holdout performance estimates real deployment quality.' },
  { term: 'Confounder', category: 'Machine Learning', level: 'Advanced', definition: 'Variable correlated with both input and target that can mislead learning.', learnMore: 'Controlling confounders improves causal validity of models.' },
  { term: 'Causal Inference', category: 'Machine Learning', level: 'Advanced', definition: 'Estimating cause-and-effect relationships from data.', learnMore: 'Causal methods help answer intervention and policy questions.' },
  { term: 'Counterfactual', category: 'Machine Learning', level: 'Advanced', definition: 'Hypothetical scenario showing what would happen under different conditions.', learnMore: 'Counterfactuals support explainability and causal analysis.' },
  { term: 'Bayesian Optimization', category: 'Machine Learning', level: 'Advanced', definition: 'Sample-efficient method for hyperparameter tuning.', learnMore: 'It models objective uncertainty to choose promising trials.' },
  { term: 'Grid Search', category: 'Machine Learning', level: 'Beginner', definition: 'Exhaustive hyperparameter search over predefined values.', learnMore: 'Simple but expensive as parameter space grows.' },
  { term: 'Random Search', category: 'Machine Learning', level: 'Beginner', definition: 'Sampling random hyperparameter combinations.', learnMore: 'Often more efficient than grid search in high dimensions.' },
  { term: 'Bayes Classifier', category: 'Machine Learning', level: 'Intermediate', definition: 'Classifier using Bayes theorem to infer class probabilities.', learnMore: 'It selects the class with highest posterior probability.' },
  { term: 'Entropy', category: 'Machine Learning', level: 'Intermediate', definition: 'Measure of uncertainty in a probability distribution.', learnMore: 'Entropy appears in information gain and cross-entropy objectives.' },
  { term: 'Information Gain', category: 'Machine Learning', level: 'Intermediate', definition: 'Reduction in entropy after a dataset split.', learnMore: 'Decision trees use information gain to choose splits.' },
  { term: 'Kernel Trick', category: 'Machine Learning', level: 'Advanced', definition: 'Computing similarity in high-dimensional space without explicit mapping.', learnMore: 'Enables nonlinear decision boundaries in SVMs.' },
  { term: 'Margin', category: 'Machine Learning', level: 'Intermediate', definition: 'Distance between decision boundary and nearest examples.', learnMore: 'Large margins often improve generalization in classifiers.' },
  { term: 'Hidden Layer', category: 'Machine Learning', level: 'Beginner', definition: 'Intermediate layer in a neural network between input and output.', learnMore: 'Hidden layers learn hierarchical representations.' },
  { term: 'Activation Function', category: 'Machine Learning', level: 'Beginner', definition: 'Nonlinear function applied to neuron outputs.', learnMore: 'Common activations include ReLU, sigmoid, and GELU.' },
  { term: 'ReLU', category: 'Machine Learning', level: 'Beginner', definition: 'Activation function max(0, x).', learnMore: 'ReLU is simple and effective for deep networks.' },
  { term: 'Sigmoid', category: 'Machine Learning', level: 'Beginner', definition: 'S-shaped activation mapping values to 0-1 range.', learnMore: 'Often used for probabilities in binary classification.' },
  { term: 'Softmax', category: 'Machine Learning', level: 'Beginner', definition: 'Function converting logits into class probabilities.', learnMore: 'Softmax outputs sum to 1 across classes.' },
  { term: 'Logits', category: 'Machine Learning', level: 'Intermediate', definition: 'Raw model scores before probability normalization.', learnMore: 'Logits are transformed by softmax or sigmoid for probabilities.' },
  { term: 'Backpropagation', category: 'Machine Learning', level: 'Intermediate', definition: 'Algorithm computing gradients through neural networks.', learnMore: 'Backprop enables efficient weight updates via chain rule.' },
  { term: 'Vanishing Gradient', category: 'Machine Learning', level: 'Advanced', definition: 'When gradients become too small to learn effectively.', learnMore: 'Careful initialization and architecture choices mitigate this issue.' },
  { term: 'Exploding Gradient', category: 'Machine Learning', level: 'Advanced', definition: 'When gradients grow excessively and destabilize training.', learnMore: 'Gradient clipping is a common mitigation strategy.' },
  { term: 'Gradient Clipping', category: 'Machine Learning', level: 'Intermediate', definition: 'Constraining gradient magnitude during training.', learnMore: 'Prevents unstable updates in deep or recurrent models.' },
  { term: 'Weight Initialization', category: 'Machine Learning', level: 'Intermediate', definition: 'Setting initial model parameters before training.', learnMore: 'Good initialization accelerates convergence and stability.' },
  { term: 'Xavier Initialization', category: 'Machine Learning', level: 'Advanced', definition: 'Initialization scheme balancing activation variance across layers.', learnMore: 'Often used for tanh/sigmoid networks.' },
  { term: 'He Initialization', category: 'Machine Learning', level: 'Advanced', definition: 'Initialization tailored for ReLU-like activations.', learnMore: 'Helps maintain stable signal flow in deep ReLU networks.' },
  { term: 'Attention Head', category: 'Generative AI', level: 'Advanced', definition: 'Parallel attention submodule within multi-head attention.', learnMore: 'Different heads can capture different relationship patterns.' },
  { term: 'Positional Encoding', category: 'Generative AI', level: 'Advanced', definition: 'Technique injecting token order information into transformers.', learnMore: 'Without it, self-attention is order-agnostic.' },
  { term: 'Masked Language Modeling', category: 'NLP', level: 'Advanced', definition: 'Pretraining task predicting masked tokens from context.', learnMore: 'Used in encoder models like BERT.' },
  { term: 'Autoregressive Modeling', category: 'Generative AI', level: 'Intermediate', definition: 'Generating each token conditioned on previous tokens.', learnMore: 'Decoder-only LLMs typically use autoregressive objectives.' },
  { term: 'Instruction Following', category: 'Generative AI', level: 'Beginner', definition: 'Model ability to obey user-provided tasks and constraints.', learnMore: 'Improved by instruction tuning and preference optimization.' },
  { term: 'Long-Context Reasoning', category: 'Generative AI', level: 'Advanced', definition: 'Ability to reason over very long inputs.', learnMore: 'Requires efficient attention and strong retrieval/organization strategies.' },
  { term: 'MoE', category: 'Generative AI', level: 'Advanced', definition: 'Mixture-of-Experts architecture activating subsets of parameters per token.', learnMore: 'MoE increases capacity while controlling per-token compute.' },
  { term: 'Routing', category: 'Generative AI', level: 'Advanced', definition: 'Selecting which experts/modules process each token or request.', learnMore: 'Effective routing is central to MoE model performance.' },
  { term: 'Speculative Decoding', category: 'Generative AI', level: 'Advanced', definition: 'Acceleration method using draft models to speed generation.', learnMore: 'Accepted draft tokens reduce expensive target-model calls.' },
  { term: 'KV Cache', category: 'Generative AI', level: 'Intermediate', definition: 'Stored key/value attention states reused during decoding.', learnMore: 'KV caching greatly improves autoregressive inference speed.' },
  { term: 'Token', category: 'Generative AI', level: 'Beginner', definition: 'Basic text unit processed by language models.', learnMore: 'Words can split into multiple subword tokens.' },
  {
    term: 'Algorithm',
    category: 'Machine Learning',
    level: 'Beginner',
    definition: 'A set of instructions a computer follows to solve a problem or make decisions.',
    learnMore:
      'In AI, algorithms define how data is processed and how predictions are produced. Different algorithms are chosen based on speed, interpretability, and accuracy needs.',
  },
  {
    term: 'Model',
    category: 'Machine Learning',
    level: 'Beginner',
    definition: 'A mathematical system trained on data to recognize patterns and make predictions.',
    learnMore:
      'After training, a model can generalize to new examples it has not seen before. Its quality depends on data quality, architecture, and tuning.',
  },
  {
    term: 'Training Data',
    category: 'Machine Learning',
    level: 'Beginner',
    definition: 'Examples used to teach a machine learning model how to perform a task.',
    learnMore:
      'Training data should represent real-world conditions; otherwise, model performance can degrade when deployed to production.',
  },
  {
    term: 'Overfitting',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'When a model learns noise in training data and performs poorly on new data.',
    learnMore:
      'Overfitting is reduced using methods such as regularization, data augmentation, early stopping, and proper validation strategies.',
  },
  {
    term: 'Underfitting',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'When a model is too simple to capture patterns in the data.',
    learnMore:
      'Underfit models show poor performance on both training and test sets. Increasing model capacity or feature quality can help.',
  },
  {
    term: 'Hyperparameter',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'A configurable setting chosen before training, such as learning rate or batch size.',
    learnMore:
      'Hyperparameters are tuned through grid search, random search, or Bayesian optimization to improve model performance.',
  },
  {
    term: 'Gradient Descent',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'An optimization method that updates model weights to reduce prediction error.',
    learnMore:
      'It works by following the slope of the loss function. Variants include stochastic, mini-batch, and momentum-based approaches.',
  },
  {
    term: 'Neural Network',
    category: 'Machine Learning',
    level: 'Beginner',
    definition: 'A model inspired by the brain, built from layers of interconnected computation units.',
    learnMore:
      'Deep neural networks can learn highly complex patterns from images, audio, and text by stacking many hidden layers.',
  },
  {
    term: 'Transformer',
    category: 'Generative AI',
    level: 'Advanced',
    definition: 'A neural architecture that uses attention to process relationships in sequences efficiently.',
    learnMore:
      'Transformers replaced many recurrent models and now power modern language models because they scale well with data and compute.',
  },
  {
    term: 'Attention Mechanism',
    category: 'Generative AI',
    level: 'Advanced',
    definition: 'A technique allowing models to focus on relevant parts of input when generating output.',
    learnMore:
      'Self-attention calculates dependencies between tokens directly, enabling better context handling in long sequences.',
  },
  {
    term: 'Large Language Model (LLM)',
    category: 'Generative AI',
    level: 'Beginner',
    definition: 'A transformer-based model trained on vast text corpora to understand and generate language.',
    learnMore:
      'LLMs can perform translation, summarization, coding, and question answering through prompting and optional fine-tuning.',
  },
  {
    term: 'Prompt Engineering',
    category: 'Generative AI',
    level: 'Intermediate',
    definition: 'Designing effective prompts to guide model behavior and output quality.',
    learnMore:
      'Strategies include role prompting, few-shot examples, constraints, and structured output formats for reliability.',
  },
  {
    term: 'Fine-Tuning',
    category: 'Generative AI',
    level: 'Advanced',
    definition: 'Further training a pre-trained model on domain-specific data for specialized tasks.',
    learnMore:
      'Fine-tuning can improve performance in narrow domains, but it requires careful evaluation to avoid overfitting or drift.',
  },
  {
    term: 'Embeddings',
    category: 'NLP',
    level: 'Intermediate',
    definition: 'Numeric vector representations that encode semantic meaning of words, sentences, or documents.',
    learnMore:
      'Similar meanings map to nearby vectors, making embeddings useful for search, recommendation, clustering, and retrieval.',
  },
  {
    term: 'Tokenization',
    category: 'NLP',
    level: 'Beginner',
    definition: 'The process of splitting text into units such as words, subwords, or symbols.',
    learnMore:
      'Tokenizer design affects model context length, speed, and multilingual performance. Subword tokenizers balance vocabulary size and coverage.',
  },
  {
    term: 'Named Entity Recognition (NER)',
    category: 'NLP',
    level: 'Intermediate',
    definition: 'Identifying and classifying entities like people, locations, and organizations in text.',
    learnMore:
      'NER supports information extraction pipelines in legal, healthcare, and customer support use cases.',
  },
  {
    term: 'Sentiment Analysis',
    category: 'NLP',
    level: 'Beginner',
    definition: 'Determining emotional tone in text, such as positive, negative, or neutral.',
    learnMore:
      'Businesses use sentiment models to monitor product feedback, social media, and support conversations at scale.',
  },
  {
    term: 'BLEU Score',
    category: 'NLP',
    level: 'Advanced',
    definition: 'A metric for evaluating machine translation quality by comparing output to reference translations.',
    learnMore:
      'BLEU estimates n-gram overlap but may miss semantic quality, so it is often combined with human or learned evaluations.',
  },
  {
    term: 'Convolutional Neural Network (CNN)',
    category: 'Computer Vision',
    level: 'Intermediate',
    definition: 'A neural network using convolution filters to detect patterns in images.',
    learnMore:
      'CNNs exploit spatial structure and have been foundational for classification, segmentation, and detection tasks.',
  },
  {
    term: 'Object Detection',
    category: 'Computer Vision',
    level: 'Intermediate',
    definition: 'Detecting objects in images and drawing bounding boxes around them.',
    learnMore:
      'Detection systems power applications like autonomous driving, retail analytics, and robotics perception.',
  },
  {
    term: 'Image Segmentation',
    category: 'Computer Vision',
    level: 'Advanced',
    definition: 'Assigning a label to each pixel in an image to separate objects and regions.',
    learnMore:
      'Segmentation is used in medical imaging, satellite mapping, and autonomous systems where precise boundaries matter.',
  },
  {
    term: 'Transfer Learning',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'Reusing a model trained on one task as a starting point for another related task.',
    learnMore:
      'Transfer learning reduces required data and compute, especially useful when labeled data is limited.',
  },
  {
    term: 'Reinforcement Learning',
    category: 'Machine Learning',
    level: 'Advanced',
    definition: 'A learning framework where agents optimize actions through rewards and penalties.',
    learnMore:
      'RL is used in game playing, robotics, and control systems where decisions impact future outcomes.',
  },
  {
    term: 'Hallucination',
    category: 'Generative AI',
    level: 'Intermediate',
    definition: 'When a model generates plausible-sounding but incorrect or fabricated information.',
    learnMore:
      'Mitigation includes retrieval grounding, confidence checks, prompt constraints, and human review loops.',
  },
  {
    term: 'Retrieval-Augmented Generation (RAG)',
    category: 'Generative AI',
    level: 'Advanced',
    definition: 'A method that retrieves external knowledge and feeds it to a model before generation.',
    learnMore:
      'RAG improves factuality and freshness by grounding responses in trusted documents rather than only model memory.',
  },
  {
    term: 'Bias',
    category: 'Ethics & Safety',
    level: 'Beginner',
    definition: 'Systematic unfairness in model outputs due to skewed data, assumptions, or design choices.',
    learnMore:
      'Bias can cause discriminatory outcomes, so audits, balanced datasets, and fairness metrics are critical.',
  },
  {
    term: 'Fairness',
    category: 'Ethics & Safety',
    level: 'Intermediate',
    definition: 'The principle that AI systems should treat individuals and groups equitably.',
    learnMore:
      'Fairness has multiple definitions (e.g., equalized odds, demographic parity), often requiring tradeoff decisions.',
  },
  {
    term: 'Explainability',
    category: 'Ethics & Safety',
    level: 'Intermediate',
    definition: 'The ability to understand and communicate why a model made a specific decision.',
    learnMore:
      'Techniques include feature importance, attention visualization, SHAP values, and model cards.',
  },
  {
    term: 'Adversarial Attack',
    category: 'Ethics & Safety',
    level: 'Advanced',
    definition: 'An intentional input manipulation designed to make a model produce wrong outputs.',
    learnMore:
      'Even tiny perturbations can fool vision models, motivating robust training and threat modeling.',
  },
  {
    term: 'Alignment',
    category: 'Ethics & Safety',
    level: 'Advanced',
    definition: 'Ensuring AI behavior matches human values, intentions, and safety constraints.',
    learnMore:
      'Alignment research addresses harmful outputs, objective misspecification, and long-term control challenges.',
  },
  {
    term: 'Data Drift',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'A shift in real-world data patterns over time compared with training data.',
    learnMore:
      'Drift monitoring is essential in production systems and often triggers retraining or pipeline updates.',
  },
  {
    term: 'Precision and Recall',
    category: 'Machine Learning',
    level: 'Intermediate',
    definition: 'Metrics that balance false positives and false negatives in classification tasks.',
    learnMore:
      'Precision asks, “Of predicted positives, how many were correct?” Recall asks, “Of true positives, how many were found?”',
  },
];

const categories = ['All', ...new Set(terms.map((item) => item.category))];
const exploredTerms = new Set();

const cardsGrid = document.getElementById('cardsGrid');
const categoryFilters = document.getElementById('categoryFilters');
const searchInput = document.getElementById('searchInput');
const progressTracker = document.getElementById('progressTracker');
const themeToggle = document.getElementById('themeToggle');
const themeLabel = document.getElementById('themeLabel');
const randomBtn = document.getElementById('randomBtn');
const emptyState = document.getElementById('emptyState');

let activeCategory = 'All';

function levelClass(level) {
  return level.toLowerCase();
}

function updateProgress(totalVisible = terms.length) {
  progressTracker.textContent = `${exploredTerms.size} of ${totalVisible} terms explored`;
}

function createCard(item) {
  const card = document.createElement('article');
  card.className = 'term-card';
  card.setAttribute('tabindex', '0');

  card.innerHTML = `
    <div class="card-inner">
      <div class="card-face card-front">
        <div class="term-head">
          <h3 class="term-name">${item.term}</h3>
          <span class="badge ${levelClass(item.level)}">${item.level}</span>
        </div>
        <p class="category">${item.category}</p>
        <p class="quiz-hint">Tap to reveal meaning</p>
      </div>
      <div class="card-face card-back">
        <div class="term-head">
          <h3 class="term-name">${item.term}</h3>
          <span class="badge ${levelClass(item.level)}">${item.level}</span>
        </div>
        <p class="category">${item.category}</p>
        <p class="definition">${item.definition}</p>
        <button class="learn-btn" type="button">Learn More</button>
        <p class="learn-more"><strong>Learn More:</strong> ${item.learnMore}</p>
      </div>
    </div>
  `;

  const learnBtn = card.querySelector('.learn-btn');

  const revealCard = () => {
    if (!card.classList.contains('revealed')) {
      card.classList.add('revealed');
      exploredTerms.add(item.term);
      updateProgress();
      return;
    }

    card.classList.toggle('expanded');
  };

  card.addEventListener('click', (event) => {
    if (event.target.classList.contains('learn-btn')) return;
    revealCard();
  });

  learnBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    card.classList.toggle('expanded');
  });

  card.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      revealCard();
    <div class="term-head">
      <h3 class="term-name">${item.term}</h3>
      <span class="badge ${levelClass(item.level)}">${item.level}</span>
    </div>
    <p class="category">${item.category}</p>
    <p class="definition">${item.definition}</p>
    <p class="learn-more"><strong>Learn More:</strong> ${item.learnMore}</p>
  `;

  const toggleExpand = () => {
    card.classList.toggle('expanded');
    exploredTerms.add(item.term);
    updateProgress();
  };

  card.addEventListener('click', toggleExpand);
  card.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleExpand();
    }
  });

  return card;
}

function renderCategories() {
  categoryFilters.innerHTML = '';
  categories.forEach((category) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `filter-btn ${category === activeCategory ? 'active' : ''}`;
    btn.textContent = category;

    btn.addEventListener('click', () => {
      activeCategory = category;
      renderCategories();
      renderCards();
    });

    categoryFilters.appendChild(btn);
  });
}

function getFilteredTerms() {
  const query = searchInput.value.trim().toLowerCase();

  return terms.filter((item) => {
    const inCategory = activeCategory === 'All' || item.category === activeCategory;
    const inSearch =
      item.term.toLowerCase().includes(query) ||
      item.definition.toLowerCase().includes(query) ||
      item.category.toLowerCase().includes(query);

    return inCategory && inSearch;
  });
}

function renderCards() {
  const filtered = getFilteredTerms();
  cardsGrid.innerHTML = '';
  filtered.forEach((item) => cardsGrid.appendChild(createCard(item)));

  filtered.forEach((item) => {
    cardsGrid.appendChild(createCard(item));
  });

  emptyState.classList.toggle('hidden', filtered.length > 0);
  updateProgress(filtered.length || 0);
}

function setTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  localStorage.setItem('theme', mode);
  themeLabel.textContent = mode === 'dark' ? '☀️ Light' : '🌙 Dark';
}

function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved) return setTheme(saved);
  if (saved) {
    setTheme(saved);
    return;
  }

  const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  setTheme(systemDark ? 'dark' : 'light');
}

function highlightRandomCard() {
  const cards = Array.from(document.querySelectorAll('.term-card'));
  if (!cards.length) return;
  const randomCard = cards[Math.floor(Math.random() * cards.length)];
  randomCard.classList.add('highlight', 'revealed');
  randomCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
  const term = randomCard.querySelector('.term-name')?.textContent;
  if (term) exploredTerms.add(term);
  updateProgress(cards.length);
  setTimeout(() => randomCard.classList.remove('highlight'), 1200);

  const randomCard = cards[Math.floor(Math.random() * cards.length)];
  randomCard.classList.add('highlight');
  randomCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

  if (!randomCard.classList.contains('expanded')) {
    randomCard.classList.add('expanded');
  }

  const term = randomCard.querySelector('.term-name')?.textContent;
  if (term) {
    exploredTerms.add(term);
    updateProgress(cards.length);
  }

  setTimeout(() => {
    randomCard.classList.remove('highlight');
  }, 1200);
}

searchInput.addEventListener('input', renderCards);
themeToggle.addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  setTheme(next);
});
randomBtn.addEventListener('click', highlightRandomCard);

initTheme();
renderCategories();
renderCards();
