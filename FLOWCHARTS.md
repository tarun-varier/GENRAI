# Music Genre Classifier - Flowcharts and Diagrams

This file contains Mermaid diagram code that can be rendered in GitHub, GitLab, or using Mermaid Live Editor (https://mermaid.live/)

---

## 1. Main Workflow Diagram - Prediction Process

```mermaid
flowchart TD
    A[Start: Audio Input] --> B{Input Type?}
    B -->|File Upload| C[Load WAV File]
    B -->|Live Recording| D[Record from Microphone]
    
    C --> E[Audio Preprocessing]
    D --> E
    
    E --> F[Load Audio with Librosa<br/>22,050 Hz, 30 seconds]
    
    F --> G[Feature Extraction]
    
    G --> H[Extract MFCCs<br/>13 coefficients × 2 stats = 26 features]
    G --> I[Extract Spectral Centroid<br/>mean + std = 2 features]
    G --> J[Extract Spectral Rolloff<br/>mean + std = 2 features]
    G --> K[Extract Zero-Crossing Rate<br/>mean + std = 2 features]
    G --> L[Extract Chroma Features<br/>12 bins × 2 stats = 24 features]
    G --> M[Extract Tempo<br/>1 feature]
    
    H --> N[Combine Features]
    I --> N
    J --> N
    K --> N
    L --> N
    M --> N
    
    N --> O[Feature Vector<br/>59 dimensions]
    
    O --> P[StandardScaler<br/>Normalization]
    
    P --> Q[Random Forest Classifier<br/>200 Decision Trees]
    
    Q --> R[Tree 1 Vote]
    Q --> S[Tree 2 Vote]
    Q --> T[Tree 3 Vote]
    Q --> U[... 197 more trees ...]
    Q --> V[Tree 200 Vote]
    
    R --> W[Aggregate Votes]
    S --> W
    T --> W
    U --> W
    V --> W
    
    W --> X[Calculate Probabilities<br/>for 10 Genres]
    
    X --> Y[Predicted Genre<br/>Highest Probability]
    
    Y --> Z[Display Results:<br/>- Genre Name<br/>- Probability Scores<br/>- Visualization]
    
    Z --> AA[End]
    
    style A fill:#e1f5ff
    style Y fill:#c8e6c9
    style Z fill:#fff9c4
    style AA fill:#ffccbc
```

---

## 2. Training Workflow Diagram

```mermaid
flowchart TD
    A[Start: GTZAN Dataset] --> B[1000 Audio Files<br/>10 Genres × 100 Files]
    
    B --> C[For Each Audio File]
    
    C --> D[Extract 59 Features]
    
    D --> E[Create Feature Matrix X<br/>Shape: 1000 × 59]
    
    E --> F[Create Label Vector y<br/>Shape: 1000 × 1]
    
    F --> G[Label Encoding<br/>Genres → 0-9]
    
    G --> H[Train-Test Split<br/>80% Train / 20% Test<br/>Stratified]
    
    H --> I[Training Set<br/>800 samples]
    H --> J[Testing Set<br/>200 samples]
    
    I --> K[Fit StandardScaler<br/>on Training Data]
    
    K --> L[Transform Training Data<br/>X_train_scaled]
    
    L --> M[Train Random Forest<br/>n_estimators=200<br/>max_depth=20]
    
    M --> N[Trained Model]
    
    J --> O[Transform Test Data<br/>using fitted scaler]
    
    O --> P[Predict on Test Set]
    
    N --> P
    
    P --> Q[Calculate Metrics:<br/>- Accuracy<br/>- Precision<br/>- Recall<br/>- F1-Score]
    
    Q --> R{Accuracy > 75%?}
    
    R -->|Yes| S[Save Model<br/>model.pkl]
    R -->|No| T[Tune Hyperparameters]
    
    T --> M
    
    S --> U[Model Ready for Deployment]
    
    U --> V[End]
    
    style A fill:#e1f5ff
    style N fill:#c8e6c9
    style S fill:#fff9c4
    style V fill:#ffccbc
```

---

## 3. Feature Extraction Pipeline

```mermaid
flowchart LR
    A[Audio Signal<br/>30 seconds] --> B[Librosa Load]
    
    B --> C{Feature Type}
    
    C -->|Timbral| D[MFCC Extraction]
    C -->|Spectral| E[Spectral Analysis]
    C -->|Harmonic| F[Chroma Extraction]
    C -->|Rhythmic| G[Tempo Detection]
    
    D --> D1[13 MFCCs]
    D1 --> D2[Calculate Mean]
    D1 --> D3[Calculate Std Dev]
    D2 --> H[26 MFCC Features]
    D3 --> H
    
    E --> E1[Spectral Centroid]
    E --> E2[Spectral Rolloff]
    E --> E3[Zero-Crossing Rate]
    E1 --> I[6 Spectral Features]
    E2 --> I
    E3 --> I
    
    F --> F1[12 Chroma Bins]
    F1 --> F2[Calculate Mean]
    F1 --> F3[Calculate Std Dev]
    F2 --> J[24 Chroma Features]
    F3 --> J
    
    G --> K[1 Tempo Feature<br/>BPM]
    
    H --> L[Concatenate All Features]
    I --> L
    J --> L
    K --> L
    
    L --> M[Feature Vector<br/>59 dimensions]
    
    style A fill:#e1f5ff
    style M fill:#c8e6c9
```

---

## 4. System Architecture Diagram

```mermaid
graph TB
    subgraph "User Interface Layer"
        A[Streamlit Web App]
        B[File Upload Component]
        C[Microphone Recorder]
        D[Visualization Dashboard]
    end
    
    subgraph "Application Layer"
        E[MusicGenreClassifier Class]
        F[Feature Extraction Module]
        G[Prediction Module]
        H[Model Management]
    end
    
    subgraph "Model Layer"
        I[Random Forest Model]
        J[StandardScaler]
        K[Label Encoder]
    end
    
    subgraph "Data Layer"
        L[GTZAN Dataset]
        M[Trained Model File<br/>model.pkl]
        N[Temporary Audio Files]
    end
    
    A --> B
    A --> C
    A --> D
    
    B --> E
    C --> E
    
    E --> F
    E --> G
    E --> H
    
    F --> I
    G --> I
    G --> J
    G --> K
    
    H --> M
    
    L --> E
    N --> F
    
    I --> D
    
    style A fill:#e3f2fd
    style E fill:#fff9c4
    style I fill:#c8e6c9
    style L fill:#ffccbc
```

---

## 5. Decision Tree Ensemble Visualization

```mermaid
flowchart TD
    A[Input: 59 Features] --> B[Random Forest<br/>200 Trees]
    
    B --> C[Tree 1]
    B --> D[Tree 2]
    B --> E[Tree 3]
    B --> F[...]
    B --> G[Tree 200]
    
    C --> C1{Feature: MFCC_1 < 0.5?}
    C1 -->|Yes| C2{Feature: Tempo < 120?}
    C1 -->|No| C3{Feature: Chroma_5 < 0.3?}
    C2 -->|Yes| C4[Vote: Classical]
    C2 -->|No| C5[Vote: Jazz]
    C3 -->|Yes| C6[Vote: Rock]
    C3 -->|No| C7[Vote: Metal]
    
    D --> D1[Vote: Hip-hop]
    E --> E1[Vote: Classical]
    F --> F1[Vote: ...]
    G --> G1[Vote: Classical]
    
    C4 --> H[Voting Pool]
    C5 --> H
    C6 --> H
    C7 --> H
    D1 --> H
    E1 --> H
    F1 --> H
    G1 --> H
    
    H --> I[Count Votes:<br/>Classical: 85<br/>Jazz: 45<br/>Rock: 30<br/>Metal: 20<br/>Hip-hop: 15<br/>Others: 5]
    
    I --> J[Winner: Classical<br/>Probability: 85/200 = 42.5%]
    
    style A fill:#e1f5ff
    style B fill:#fff9c4
    style H fill:#ffccbc
    style J fill:#c8e6c9
```

---

## 6. Real-time Prediction Flow

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant C as Classifier
    participant L as Librosa
    participant M as Model
    
    U->>UI: Click Record Button
    UI->>UI: Start Microphone Recording
    U->>UI: Play Music (10-30 sec)
    U->>UI: Stop Recording
    UI->>UI: Save Audio to Temp File
    UI->>C: predict(audio_file)
    C->>L: Load Audio
    L-->>C: Audio Array
    C->>L: Extract Features
    L-->>C: 59 Features
    C->>C: Scale Features
    C->>M: Predict
    M-->>C: Probabilities [10 genres]
    C-->>UI: Genre + Probabilities
    UI->>UI: Generate Visualization
    UI-->>U: Display Results
    UI->>UI: Delete Temp File
```

---

## 7. Genre Classification Decision Boundaries

```mermaid
graph TD
    A[Audio Features] --> B{High Tempo<br/>> 140 BPM?}
    
    B -->|Yes| C{High Spectral<br/>Centroid?}
    B -->|No| D{Low Zero-Crossing<br/>Rate?}
    
    C -->|Yes| E[Metal / Disco]
    C -->|No| F[Hip-hop / Pop]
    
    D -->|Yes| G{Complex<br/>Chroma?}
    D -->|No| H{Guitar-heavy<br/>MFCCs?}
    
    G -->|Yes| I[Classical / Jazz]
    G -->|No| J[Blues]
    
    H -->|Yes| K[Rock / Country]
    H -->|No| L[Reggae]
    
    E --> M[Final Classification<br/>via Random Forest]
    F --> M
    I --> M
    J --> M
    K --> M
    L --> M
    
    style A fill:#e1f5ff
    style M fill:#c8e6c9
```

---

## How to Use These Diagrams

### Option 1: Mermaid Live Editor (Recommended)
1. Go to https://mermaid.live/
2. Copy any diagram code from above
3. Paste into the editor
4. Export as PNG, SVG, or PDF

### Option 2: GitHub/GitLab
- These diagrams render automatically in GitHub and GitLab markdown files

### Option 3: VS Code
- Install "Markdown Preview Mermaid Support" extension
- View this file in preview mode

### Option 4: Draw.io / Lucidchart
- Use the text descriptions to manually create diagrams
- Import Mermaid diagrams (some tools support this)

### Option 5: Miro Board
To create in Miro:
1. Create shapes for each box
2. Connect with arrows
3. Use colors from the style definitions:
   - #e1f5ff (light blue) - Input/Start
   - #c8e6c9 (light green) - Output/Success
   - #fff9c4 (light yellow) - Processing
   - #ffccbc (light orange) - End/Decision

---

## Diagram Descriptions for Manual Creation

If you prefer to create diagrams manually in Miro or other tools, here are the key components:

### Main Workflow (Top to Bottom):
1. **Input Layer**: Audio File Upload OR Microphone Recording
2. **Preprocessing**: Load audio, normalize
3. **Feature Extraction**: 5 parallel branches (MFCCs, Spectral, Chroma, ZCR, Tempo)
4. **Feature Combination**: Merge into 59-dimensional vector
5. **Normalization**: StandardScaler
6. **Classification**: Random Forest (200 trees)
7. **Output**: Genre prediction + probabilities

### Training Workflow (Top to Bottom):
1. **Data Loading**: GTZAN Dataset (1000 files)
2. **Feature Extraction**: Process all files
3. **Data Splitting**: 80/20 train/test
4. **Scaling**: Fit scaler on training data
5. **Training**: Random Forest training
6. **Evaluation**: Calculate metrics
7. **Model Saving**: Save to pickle file

Use rectangles for processes, diamonds for decisions, and arrows for flow direction.
