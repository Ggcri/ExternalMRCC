flowchart TD
    %% --- DEFINIZIONE STILI ---
    classDef gaussian fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:black
    classDef central fill:#e1f5fe,stroke:#0277bd,stroke-width:3px,color:black
    classDef parallel fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:black
    classDef symmetry fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:black
    classDef external fill:#fbe9e7,stroke:#c62828,stroke-width:2px,color:black
    classDef file fill:#ffffff,stroke:#333,stroke-dasharray: 5 5,color:black

    %% --- LIVELLO GAUSSIAN ---
    G_Driver("Gaussian Driver
    ( External call)")
    
    F_EIn("File: .EIn
    (Geom in a.u.)")
    
    F_EOut("File: .EOut
    (Grad in Hartree/Bohr)")

    %% --- CENTRAL DISPATCHER ---
    Central("CentralExt
    (Main Dispatcher)")
    
    %% Decisione: Parallelo o Seriale?
    IsParall{"Mode:
    Parallel?"}

    %% --- RAMO SERIALE ---
    SerialExec("Direct Execution
    subprocess.run()")

    %% --- RAMO PARALLELO (elecext.parall) ---
    subgraph ParallelMgr [Parallel Execution Manager]
        direction TB
        
        FakeFreq("1. Fake Freq Run
        (Gaussian freq=num
        Generates Displacements)")
        
        Recipe("2. Recipe Parser
        Map: TaskID -> Geom")
        
        Pool("3. ThreadPoolExecutor
        (Scalable Workers)")
        
        subgraph TaskWorker [Single Task Execution]
            WorkerScript("Interface Script
            (*Ext.py)")
            
            Binary("External Binary
            (MRCC, Molpro, Gaussian)")
            
            WorkerScript -->|Writes Input| Binary
            Binary -->|Reads Output| WorkerScript
        end
    end

    %% --- SYMMETRY ENGINE ---
    subgraph SymmEngine [Symmetry Engine]
        direction TB
        
        LogParse("Log Parser
        Extract Point Group
        & Operations R and P")
        
        VectorMap("Vector Mapping
        Grad_tgt = R * Grad_src")
        
        Assembler("Gradient Assembly
        Combine Calculated +
        Symmetry Derived")
        
        LogParse --> VectorMap --> Assembler
    end

    %% --- FLUSSO LOGICO ---

    %% 1. Avvio
    G_Driver -->|Writes| F_EIn
    F_EIn --> Central
    Central --> IsParall

    %% 2. Branching
    IsParall -- No --> SerialExec
    IsParall -- Yes --> FakeFreq

    %% 3. Parallel Loop
    FakeFreq -->|Log Analysis| Recipe
    Recipe -->|Task List| Pool
    Pool -->|Spawns| WorkerScript
    
    %% 4. Ritorno Dati Paralleli
    TaskWorker -->|Return Energy| Pool
    Pool -->|List of Energies| Assembler

    %% 5. Integrazione Simmetria
    FakeFreq -.->|Log Info| LogParse
    Assembler -->|Full Gradient| Central

    %% 6. Ritorno Seriale
    SerialExec -->|Full Gradient| Central

    %% 7. Output Finale
    Central -->|Writes| F_EOut
    F_EOut -->|Reads| G_Driver

    %% --- ASSEGNAZIONE CLASSI ---
    class G_Driver gaussian
    class F_EIn,F_EOut file
    class Central central
    class FakeFreq,Recipe,Pool parallel
    class LogParse,VectorMap,Assembler symmetry
    class SerialExec,WorkerScript,Binary external
