```
┌──────────────────────────────────────────────────────────────────┐ 
│                    SkinSegmentSrgan System                       │ 
├─────────────────┬─────────────────┬─────────────────┬────────────┤ 
│   Data Input    │   Segmentation  │ Super-Resolution│ Integration│ 
│     Module      │     Module      │     Module      │   Module   │ 
├─────────────────┼─────────────────┼─────────────────┼────────────┤ 
│ • Image Loading │ • U-Net Model   │ • SRGAN Model   │ • Pipeline │ 
│ • Validation    │ • Preprocessing │ • Enhancement   │ • Quality  │ 
│ • Preprocessing │ • Inference     │ • Postprocess   │ • Output   │ 
│ • Metadata      │ • Postprocess   │ • Validation    │ • Logging  │ 
└─────────────────┴───────────────────┴───────────────────┴────────┘ 
         │                   │                   │           │       
         ▼                   ▼                   ▼           ▼       
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌───────┐
│  Configuration  │ │   Model Store   │ │   Monitoring    │ │ Utils │
│   Management    │ │   & Versioning  │ │   & Logging     │ │ & Viz │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └───────┘ 
```

```mermaid
flowchart TD
    A[Data Input Module] --> B[Segmentation Module]
    A --> C[Super-Resolution Module]
    B --> D[Integration Module]
    C --> D

    subgraph Data Flow
        A -->|DICOM/JPG/PNG| B
        A -->|Low-Res Images| C
        B -->|Segmentation Masks| D
        C -->|Enhanced Images| D
    end

    subgraph Support Modules
        E[Configuration Management]
        F[Model Store & Versioning]
        G[Monitoring & Logging]
        H[Utils & Visualization]
    end

    D --> E
    D --> F
    D --> G
    D --> H

    classDef module fill:#2e4053,stroke:#f8f9fa,color:white;
    classDef support fill:#5d6d7e,stroke:#f8f9fa,color:white;
    class A,B,C,D module;
    class E,F,G,H support;
```