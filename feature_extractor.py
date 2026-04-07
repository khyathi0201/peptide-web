import numpy as np
from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
GAAC_GROUPS = {
    "Aliphatic" : list("GAVLMI"),
    "Aromatic"  : list("FYW"),
    "Positive"  : list("KRH"),
    "Negative"  : list("DE"),
    "Uncharged" : list("STCPNQ"),
}
KD_HYDRO = {
    "A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,
    "Q":-3.5,"E":-3.5,"G":-0.4,"H":-3.2,"I":4.5,
    "L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,
    "S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2,
}
SEQ_GROUPS = {
    "Hydrophobic": list("ACFGILMVWY"),
    "Hydrophilic": list("DEHKRNPQST"),
}
GROUP_PAIRS = list(product(SEQ_GROUPS.keys(), repeat=2))

def clean_sequence(seq):
    return "".join(aa for aa in str(seq).upper() if aa in AMINO_ACIDS)

def compute_aac(seq):
    n = len(seq)
    if n == 0: return np.zeros(20)
    return np.array([seq.count(aa)/n for aa in AMINO_ACIDS])

def compute_gaac(seq):
    n = len(seq)
    if n == 0: return np.zeros(5)
    return np.array([sum(seq.count(aa) for aa in m)/n for m in GAAC_GROUPS.values()])

def compute_asdc(seq):
    n = len(seq); vec = np.zeros(400)
    if n < 2: return vec
    pairs = {f"{a}{b}":idx for idx,(a,b) in enumerate(product(AMINO_ACIDS,repeat=2))}
    for k in range(n-1):
        denom = n-1-k
        if denom <= 0: break
        for i in range(n-1-k):
            dp = seq[i]+seq[i+k+1]
            if dp in pairs: vec[pairs[dp]] += 1.0/denom
    if (n-1)>0: vec /= (n-1)
    return vec

def compute_physico(seq):
    if len(seq)==0: return np.zeros(29)
    try:
        pa=ProteinAnalysis(seq)
        mw=pa.molecular_weight(); gravy=pa.gravy()
        instability=pa.instability_index(); aliphatic=pa.aliphatic_index()
        charge_7=pa.charge_at_pH(7.0); charge_55=pa.charge_at_pH(5.5); charge_9=pa.charge_at_pH(9.0)
        try: isoelec=pa.isoelectric_point()
        except: isoelec=7.0
        length=len(seq); aromaticity=pa.aromaticity()
        helix,turn,sheet=pa.secondary_structure_fraction()
        hv=np.array([KD_HYDRO.get(aa,0.0) for aa in seq])
        pos_set=set("KRH"); neg_set=set("DE")
        pos_count=sum(seq.count(aa) for aa in "KRH")
        neg_count=sum(seq.count(aa) for aa in "DE")
        pos_pairs=sum(1 for i in range(length-1) if seq[i] in pos_set and seq[i+1] in pos_set)
        neg_pairs=sum(1 for i in range(length-1) if seq[i] in neg_set and seq[i+1] in neg_set)
        return np.array([
            mw,gravy,instability,aliphatic,charge_7,charge_55,charge_9,isoelec,
            float(length),aromaticity,helix,turn,sheet,
            hv.mean(),hv.std(),hv.min(),hv.max(),hv.max()-hv.min(),
            pos_count,neg_count,pos_count/length,neg_count/length,float(pos_count-neg_count),
            seq.count("C")/length,seq.count("P")/length,seq.count("G")/length,
            pos_pairs,neg_pairs,float(pos_pairs+neg_pairs)
        ],dtype=np.float64)
    except: return np.zeros(29)

def group_of(aa):
    return "Hydrophobic" if aa in SEQ_GROUPS["Hydrophobic"] else "Hydrophilic"

def compute_seqpat(seq):
    n=len(seq); feats=[]
    for k in range(10):
        denom=max(n-k-1,1)
        pc={(g1,g2):0 for g1,g2 in GROUP_PAIRS}
        for i in range(n-k-1):
            if i+k+1<n:
                pc[(group_of(seq[i]),group_of(seq[i+k+1]))]+=1
        for g1,g2 in GROUP_PAIRS:
            cnt=pc[(g1,g2)]; feats.extend([float(cnt),cnt/denom])
    return np.array(feats,dtype=np.float64)

def extract_features(seq):
    """
    Main function — call this from your website backend.
    Input : peptide sequence string
    Output: numpy array of shape (534,)
    """
    seq = clean_sequence(seq)
    vec = np.concatenate([
        compute_aac(seq),     # 20
        compute_gaac(seq),    #  5
        compute_asdc(seq),    # 400
        compute_physico(seq), # 29
        compute_seqpat(seq),  # 80
    ])                        # = 534
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# Quick test
if __name__ == "__main__":
    test = extract_features("ACDEFGHIKLMNPQRSTVWY")
    print(f"Feature vector size: {len(test)}  (expected 534)")
    assert len(test) == 534
    print("All good!")
