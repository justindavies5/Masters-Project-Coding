# All in one pipeline combining everything to get up to the analaysis stage. 
# Goes into every species folder in johanna_genomes, extracts the reference and annotated reference files,
# runs the singleton isolation, runs the recombination masking, and outputs into respective subdirectories in my directory
# where each folder is for a respective file type and contains every species.
# After running this code, I ended up with the exact same singleton-isolated, recombination-masked vcfs as Maisie and Eleanor.
# If there are differences between us in the results, it will be after this. 
from pathlib import Path
from collections import defaultdict
import shutil

#Settings, main directory with all the raw 
BASE_DIR = Path("/shared/team/2025-masters-project/people/johanna_genomes/")

#Directories for outputs
SINGLETON_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_singletons")
WINDOWMASKED_SINGLETON_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_singletonsandwindowmasked")
ANNOT_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_referenceandFFA_files")
RAW_WINDOWMASKED_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_windowmasked")

#To create new directories if necessary
SINGLETON_DIR.mkdir(parents=True, exist_ok=True)
WINDOWMASKED_SINGLETON_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_DIR.mkdir(parents=True, exist_ok=True)
RAW_WINDOWMASKED_DIR.mkdir(parents=True, exist_ok=True)

# Set to True to preview actions without writing files
DRY_RUN = False

# Basic SNP bases
bases = {"A", "C", "G", "T"}

# Optional sanity check
if not BASE_DIR.exists():
    raise FileNotFoundError(f"BASE_DIR does not exist: {BASE_DIR}")

#Helper functions
def expected_paths(species_name: str):
    return {
        "singleton_vcf": SINGLETON_DIR / f"{species_name}_biallelic_k1_singletons.vcf",
        "windowmasked_singleton_vcf": WINDOWMASKED_SINGLETON_DIR / f"{species_name}_biallelic_k1_singletons_windowmasked.vcf",
        "copied_gff": ANNOT_DIR / f"{species_name}.gff",
        "copied_fna": ANNOT_DIR / f"{species_name}.fna",
        "raw_windowmasked_vcf": RAW_WINDOWMASKED_DIR / f"{species_name}_parsnp_windowmasked.vcf",
    }

def write_singleton_vcf(vcf_in: Path, vcf_out: Path):
    kept = 0
    seen = 0

    if DRY_RUN:
        print(f"[DRY RUN] Would write singleton VCF: {vcf_out}")
        return seen, kept

    with open(vcf_in, "r") as fin, open(vcf_out, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue

            seen += 1
            col = line.rstrip("\n").split("\t")

            # PASS only
            if col[6] != "PASS":
                continue

            ref = col[3]
            alt_field = col[4]

            # Exclude multiallelic
            if "," in alt_field:
                continue

            alt = alt_field

            # SNP-only
            if ref not in bases or alt not in bases:
                continue

            # FORMAT must include GT
            format_fields = col[8].split(":")
            if "GT" not in format_fields:
                continue
            gt_index = format_fields.index("GT")

            carriers = 0

            # Count carriers across all sample columns
            for sample in col[9:]:
                fields = sample.split(":")
                if gt_index >= len(fields):
                    continue

                gt = fields[gt_index]
                if gt in {".", "./.", ".|."}:
                    continue

                alleles = gt.replace("|", "/").split("/")

                # Carrier definition: sample has ALT allele at least once
                if "1" in alleles:
                    carriers += 1
                    if carriers > 1:
                        break

            # Keep only k=1
            if carriers == 1:
                fout.write(line)
                kept += 1

    return seen, kept


#Recombination masking code

def recombination_check_master(master_file, kbgap=2000, min_snps=3):

    headers = []
    data = []

    with open(master_file, "r") as inp:
        for line in inp:
            if line.startswith("#"):
                headers.append(line.rstrip("\n"))
            else:
                data.append(line.strip().split("\t"))

    columns = headers[-1].split("\t")
    genomes = columns[9:]

    recombination_list = []

    for genome in genomes:
        col_idx = columns.index(genome)

        snp_positions = defaultdict(list)

        # Collect SNP positions for this genome
        for row in data:
            if row[col_idx] == "1":
                chrom = row[0]
                pos = int(row[1])
                snp_positions[chrom].append(pos)

        # Sliding window per chromosome
        for chrom, positions in snp_positions.items():
            positions.sort()

            left = 0
            for right in range(len(positions)):
                
                # Shrink window if too large
                while positions[right] - positions[left] > kbgap:
                    left += 1

                # Check window size
                if right - left + 1 >= min_snps:
                    for i in range(left, right + 1):
                        recombination_list.append([chrom, positions[i], genome])

    # Remove duplicates
    seen = set()
    unique = []
    for r in recombination_list:
        t = tuple(r)
        if t not in seen:
            seen.add(t)
            unique.append(r)

    print(f"Detected {len(unique)} recombination SNPs")
    return unique


def clean_master_vcf(master_file, output_file, recombination_list):

    headers = []
    data = []
    changes = 0

    with open(master_file, "r") as inp:
        for line in inp:
            if line.startswith("#"):
                headers.append(line.rstrip("\n"))
            else:
                data.append(line.strip().split("\t"))

    columns = headers[-1].split("\t")
    col_idx = {col: i for i, col in enumerate(columns)}
    row_idx = {(row[0], str(row[1])): i for i, row in enumerate(data)}

    for chrom, base, genome in recombination_list:
        r = row_idx.get((chrom, str(base)))
        c = col_idx.get(genome)

        if r is not None and c is not None:
            if data[r][c] == "1":
                data[r][c] = "0"
                changes += 1

    if DRY_RUN:
        print(f"[DRY RUN] Would write windowmasked VCF: {output_file}")
        print(f"[DRY RUN] Total SNPs changed: {changes}")
        return

    with open(output_file, "w") as out:
        for h in headers:
            out.write(h + "\n")
        for row in data:
            out.write("\t".join(row) + "\n")

    print(f"Wrote {output_file}")
    print(f"Total SNPs changed: {changes}")

# Copy GFF and FNA files

def copy_annotation_files(species_dir: Path, species_name: str, out_paths: dict):
    gff_files = list(species_dir.glob("*.gff"))

    ref_fasta_dir = species_dir / "reference_fasta"
    fna_files = list(ref_fasta_dir.glob("*.fna")) if ref_fasta_dir.exists() else []

    if len(gff_files) != 1:
        print(f"  Skipping annotation copy for {species_name}: expected 1 .gff file, found {len(gff_files)}")
        return False

    if len(fna_files) != 1:
        print(f"  Skipping annotation copy for {species_name}: expected 1 .fna file in reference_fasta, found {len(fna_files)}")
        return False

    gff_src = gff_files[0]
    fna_src = fna_files[0]

    gff_dst = out_paths["copied_gff"]
    fna_dst = out_paths["copied_fna"]

    if DRY_RUN:
        print(f"[DRY RUN] Would copy {gff_src} -> {gff_dst}")
        print(f"[DRY RUN] Would copy {fna_src} -> {fna_dst}")
        return True

    shutil.copy2(gff_src, gff_dst)
    shutil.copy2(fna_src, fna_dst)

    print(f"  Copied GFF: {gff_dst.name}")
    print(f"  Copied FNA: {fna_dst.name}")
    return True

# Main pipeline

processed_species = 0
skipped_species = 0

step_counts = {
    "singleton_written": 0,
    "singleton_windowmasked": 0,
    "annotations_copied": 0,
    "raw_windowmasked": 0,
}

for species_dir in sorted(BASE_DIR.iterdir()):
    if not species_dir.is_dir():
        continue

    species_name = species_dir.name
    raw_vcf = species_dir / "parsnp_results" / "parsnp.vcf"

    # Only process species with parsnp_results/parsnp.vcf
    if not raw_vcf.exists():
        skipped_species += 1
        continue

    out_paths = expected_paths(species_name)

    print(f"\n Processing {species_name} ")
    print(f"Raw input: {raw_vcf}")

 # 1. Singleton isolation
    singleton_vcf = out_paths["singleton_vcf"]

    if singleton_vcf.exists():
        print(f"  Singleton VCF already exists: {singleton_vcf.name}")
    else:
        seen, kept = write_singleton_vcf(raw_vcf, singleton_vcf)
        print(f"  Wrote singleton VCF: {singleton_vcf.name}")
        print(f"  Variant rows seen: {seen}")
        print(f"  Singleton rows kept: {kept}")
        step_counts["singleton_written"] += 1

 # 2. Recombination masking the singleton isolated vcf
    windowmasked_singleton_vcf = out_paths["windowmasked_singleton_vcf"]

    if windowmasked_singleton_vcf.exists():
        print(f"  Windowmasked singleton VCF already exists: {windowmasked_singleton_vcf.name}")

    elif not singleton_vcf.exists():
        print(f"  Skipping windowmasking (singleton VCF not present yet): {singleton_vcf.name}")

    else:
        recombination_sites = recombination_check_master(
            singleton_vcf,
            kbgap=2000,
            min_snps=3
        )
        clean_master_vcf(
            singleton_vcf,
            windowmasked_singleton_vcf,
            recombination_list=recombination_sites
        )
        step_counts["singleton_windowmasked"] += 1

 # 3. Copy GFF and FNA files
    if out_paths["copied_gff"].exists() and out_paths["copied_fna"].exists():
        print("  Annotation files already copied")
    else:
        copied_ok = copy_annotation_files(species_dir, species_name, out_paths)
        if copied_ok:
            step_counts["annotations_copied"] += 1

# 4. Recombination masking raw parsnp vcf (for allele frequency analysis)
    raw_windowmasked_vcf = out_paths["raw_windowmasked_vcf"]

    if raw_windowmasked_vcf.exists():
        print(f"  Raw windowmasked VCF already exists: {raw_windowmasked_vcf.name}")

    elif not raw_vcf.exists():
        print(f"  Skipping raw windowmasking (raw VCF missing): {raw_vcf.name}")

    else:
        recombination_sites = recombination_check_master(
            raw_vcf,
            kbgap=2000,
            min_snps=3
        )
        clean_master_vcf(
            raw_vcf,
            raw_windowmasked_vcf,
            recombination_list=recombination_sites
        )
        step_counts["raw_windowmasked"] += 1

    processed_species += 1


#Final summary

print("\nSummary")
print(f"Species with parsnp_results/parsnp.vcf found: {processed_species}")
print(f"Species skipped (no Parsnp VCF):              {skipped_species}")
print("")
print(f"Singleton VCFs written:                       {step_counts['singleton_written']}")
print(f"Windowmasked singleton VCFs written:          {step_counts['singleton_windowmasked']}")
print(f"Annotation file sets copied:                  {step_counts['annotations_copied']}")
print(f"Raw windowmasked VCFs written:                {step_counts['raw_windowmasked']}")