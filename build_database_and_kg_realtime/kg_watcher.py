"""
kg_watcher.py — Real-Time Knowledge Graph Sync
================================================
Watches profiles.csv for changes and automatically:
  1. Detects new/updated/deleted rows via SHA-256 hashing
  2. Extracts KG entities using Qwen3 8B (3-pass pipeline)
  3. Pushes to Neo4j with clean updates (no stale data)
  4. Updates reference .md files with new entities
  5. Removes deleted candidates from Neo4j

Usage:
    conda activate mfadenv
    python kg_watcher.py
"""

import ollama
import json
import re
import os
import hashlib
import time
import pandas as pd
from typing import Optional, List
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, Field, field_validator
from thefuzz import fuzz
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

CSV_FILE = 'profiles.csv'
CHECKPOINT_FILE = 'kg_checkpoint.json'
DEBOUNCE_SECONDS = 3  # Wait 3s after last save before processing
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds (doubles each retry)

# ══════════════════════════════════════════════════════════════
# CONNECTIONS
# ══════════════════════════════════════════════════════════════

load_dotenv()
URI  = os.getenv('NEO4J_URI')
AUTH = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
DB   = os.getenv('NEO4J_DATABASE')

driver = GraphDatabase.driver(URI, auth=AUTH)

# ══════════════════════════════════════════════════════════════
# PYDANTIC MODELS (matches KG_schema.md exactly)
# ══════════════════════════════════════════════════════════════

VALID_PROFICIENCY = {'beginner', 'advanced beginner', 'intermediate', 'competent', 'proficient', 'expert'}
VALID_IMPORTANCE  = {'high', 'medium', 'low'}
VALID_SKILL_TYPE  = {'core', 'secondary', 'soft'}
VALID_SENIORITY   = {'intern', 'trainee', 'junior', 'associate', 'specialist', 'consultant',
                     'senior', 'lead', 'manager', 'principal', 'director', 'executive', 'chief'}

class SkillNode(BaseModel):
    name: str
    category: Optional[str] = None
    aliases: Optional[List[str]] = []
    @field_validator('name')
    @classmethod
    def normalize_name(cls, v): return v.strip()

class RoleNode(BaseModel):
    name: str
    description: Optional[str] = None

class DomainNode(BaseModel):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None

class SubDomainNode(BaseModel):
    name: str
    core_focus: Optional[str] = None

class CandidateNode(BaseModel):
    id: str
    name: str
    experience: Optional[float] = None

class HasSkillRel(BaseModel):
    candidate: str
    skill: str
    proficiency_level: Optional[str] = None
    type: Optional[str] = None
    @field_validator('proficiency_level')
    @classmethod
    def validate_prof(cls, v):
        if v and v.lower() not in VALID_PROFICIENCY: return None
        return v.lower() if v else None
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v and v.lower() not in VALID_SKILL_TYPE: return None
        return v.lower() if v else None

class SuitableForRel(BaseModel):
    candidate: str
    role: str
    level: Optional[str] = None
    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        if v and v.lower() not in VALID_SENIORITY: return None
        return v.lower() if v else None

class RequiresRel(BaseModel):
    role: str
    skill: str
    importance: Optional[str] = None
    min_level: Optional[str] = None
    @field_validator('importance')
    @classmethod
    def validate_imp(cls, v):
        if v and v.lower() not in VALID_IMPORTANCE: return None
        return v.lower() if v else None

class BelongsToRel(BaseModel):
    skill: str
    role: str
    is_core: Optional[bool] = None

class HasSubdomainRel(BaseModel):
    domain: str
    subdomain: str

class HasRoleRel(BaseModel):
    subdomain: str
    role: str
    priority: Optional[float] = Field(None, ge=0, le=1)

class RelatedToRel(BaseModel):
    skill_from: str
    skill_to: str
    type: Optional[str] = None
    weight: Optional[float] = Field(None, ge=0, le=1)

class KGExtraction(BaseModel):
    Domains: Optional[List[DomainNode]] = []
    SubDomains: Optional[List[SubDomainNode]] = []
    Roles: Optional[List[RoleNode]] = []
    Skills: Optional[List[SkillNode]] = []
    Candidates: Optional[List[CandidateNode]] = []
    HAS_SUBDOMAIN: Optional[List[HasSubdomainRel]] = []
    HAS_ROLE: Optional[List[HasRoleRel]] = []
    BELONGS_TO: Optional[List[BelongsToRel]] = []
    REQUIRES: Optional[List[RequiresRel]] = []
    HAS_SKILL: Optional[List[HasSkillRel]] = []
    SUITABLE_FOR: Optional[List[SuitableForRel]] = []
    RELATED_TO: Optional[List[RelatedToRel]] = []

# ══════════════════════════════════════════════════════════════
# REFERENCE FILE LOADING
# ══════════════════════════════════════════════════════════════

def load_md_list(filepath, pattern=r'^\*\s+(.+)$'):
    items = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(pattern, line.strip())
                if m: items.add(m.group(1).strip())
    except FileNotFoundError:
        pass
    return items

def load_md_headings(filepath, level='##'):
    items = set()
    prefix = level + ' '
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(prefix) and not stripped.startswith(level + '#'):
                    name = stripped[len(prefix):].strip()
                    name = re.sub(r'\s*\(\d+\s+skills\)\s*-?$', '', name).strip()
                    if name: items.add(name)
    except FileNotFoundError:
        pass
    return items

def load_skills_from_md(filepath):
    skills = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('* '):
                    raw = stripped[2:].strip()
                    clean = re.sub(r'\s*\([^)]*\)\s*$', '', raw).strip()
                    if clean and len(clean) >= 2: skills.add(clean)
    except FileNotFoundError:
        pass
    return skills

def load_all_references():
    """Load all reference .md files and return normalized maps."""
    known_domains    = load_md_headings('domains.md', '##')
    known_subdomains = load_md_headings('subdomains_and _roles.md', '###')
    known_roles      = load_md_list('subdomains_and _roles.md')
    known_skills     = load_skills_from_md('skills_and_role.md')
    
    return {
        'norm_domains':    {d.lower(): d for d in known_domains},
        'norm_subdomains': {s.lower(): s for s in known_subdomains},
        'norm_roles':      {r.lower(): r for r in known_roles},
        'norm_skills':     {s.lower(): s for s in known_skills},
        'new_domains':     set(),
        'new_subdomains':  set(),
        'new_roles':       set(),
        'new_skills':      set(),
    }

# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

def load_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    except: return '(file not found)'

def build_system_prompt():
    schema_text     = load_file('KG_schema.md')
    domains_text    = load_file('domains.md')
    levels_text     = load_file('extracted_levels.md')
    subdomains_text = load_file('subdomains_and _roles.md')
    
    return f"""You are an expert Knowledge Graph entity extractor. Your job is to extract structured graph data from a candidate profile.

=== STRICT SCHEMA (follow exactly) ===
{schema_text}

=== EXISTING DOMAINS ===
{domains_text}

=== EXISTING SUBDOMAINS & ROLES ===
{subdomains_text}

=== VALID PROFICIENCY & SENIORITY LEVELS ===
{levels_text}

=== CRITICAL RULES ===
1. Output PURE JSON only. No markdown fences, no commentary.
2. The JSON must have these top-level keys (arrays): "Domains", "SubDomains", "Roles", "Skills", "Candidates", "HAS_SUBDOMAIN", "HAS_ROLE", "BELONGS_TO", "REQUIRES", "HAS_SKILL", "SUITABLE_FOR", "RELATED_TO"
3. Node types allowed: Domain, SubDomain, Role, Skill, Candidate. NO other node types.
4. Relationship types allowed: HAS_SUBDOMAIN, HAS_ROLE, BELONGS_TO, REQUIRES, HAS_SKILL, SUITABLE_FOR, RELATED_TO. NO other relationship types.
5. PREFER existing domains/subdomains/roles from the lists above. Only create new ones if truly necessary.
6. Normalize skill names: "Python Programming" → "Python", "python" → "Python". Store variants in aliases.
7. Proficiency levels on HAS_SKILL must be one of: beginner, advanced beginner, intermediate, competent, proficient, expert
8. Skill type on HAS_SKILL must be: core, secondary, or soft
9. If any property is unknown, set it to null.
10. Extract RELATED_TO relationships between skills where logical (e.g., Python RELATED_TO Django, type: "built_on")
11. BELONGS_TO connects Skill → Role (NOT SubDomain).

=== OUTPUT FORMAT EXAMPLE ===
{{
  "Domains": [{{"name": "Engineering, IT & Software", "description": "...", "category": "Technical"}}],
  "SubDomains": [{{"name": "Backend Development", "core_focus": "Server-side logic"}}],
  "Roles": [{{"name": "Backend Developer", "description": "Builds server-side applications"}}],
  "Skills": [{{"name": "Python", "category": "language", "aliases": ["Python Programming", "Py"]}}],
  "Candidates": [{{"id": "12345", "name": "John", "experience": 5.0}}],
  "HAS_SUBDOMAIN": [{{"domain": "Engineering, IT & Software", "subdomain": "Backend Development"}}],
  "HAS_ROLE": [{{"subdomain": "Backend Development", "role": "Backend Developer", "priority": 0.9}}],
  "BELONGS_TO": [{{"skill": "Python", "role": "Backend Developer", "is_core": true}}],
  "REQUIRES": [{{"role": "Backend Developer", "skill": "Python", "importance": "high", "min_level": "competent"}}],
  "HAS_SKILL": [{{"candidate": "12345", "skill": "Python", "proficiency_level": "expert", "type": "core"}}],
  "SUITABLE_FOR": [{{"candidate": "12345", "role": "Backend Developer", "level": "senior"}}],
  "RELATED_TO": [{{"skill_from": "Python", "skill_to": "Django", "type": "built_on", "weight": 0.9}}]
}}
"""

# ══════════════════════════════════════════════════════════════
# ROW → PARAGRAPH
# ══════════════════════════════════════════════════════════════

def row_to_paragraph(row):
    cid   = str(row.get('id', 'unknown')).strip()
    name  = str(row.get('name', '')).strip() or f'Candidate_{cid}'
    exp   = row.get('years_of_experience', 'N/A')
    core  = str(row.get('core_skills', '')).strip()
    sec   = str(row.get('secondary_skills', '')).strip()
    soft  = str(row.get('soft_skills', '')).strip()
    roles = str(row.get('potential_roles', '')).strip()
    summary = str(row.get('skill_summary', '')).strip()
    
    return f"""Candidate '{name}' (ID: {cid}) has {exp} years of experience.
Core Skills: {core if core else 'None listed'}
Secondary Skills: {sec if sec else 'None listed'}
Soft Skills: {soft if soft else 'None listed'}
Potential Roles: {roles if roles else 'None listed'}
Skill Summary: {summary if summary else 'No summary available'}"""

# ══════════════════════════════════════════════════════════════
# PASS 1: LLM EXTRACTION
# ══════════════════════════════════════════════════════════════

def pass1_extract(paragraph, system_prompt):
    response = ollama.chat(
        model='qwen2.5:7b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'Extract the Knowledge Graph entities from this candidate profile:\n\n{paragraph}'}
        ],
        options={'num_gpu': 99}
    )
    raw = response['message']['content']
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f'    ⚠ JSON parse error: {e}')
        return {}

# ══════════════════════════════════════════════════════════════
# PASS 2: PYDANTIC VALIDATION
# ══════════════════════════════════════════════════════════════

def pass2_normalize(raw_json):
    if not raw_json: return KGExtraction()
    def safe_parse(model_cls, items):
        results = []
        for item in (items or []):
            try: results.append(model_cls(**item))
            except: pass
        return results
    return KGExtraction(
        Domains=safe_parse(DomainNode, raw_json.get('Domains', [])),
        SubDomains=safe_parse(SubDomainNode, raw_json.get('SubDomains', [])),
        Roles=safe_parse(RoleNode, raw_json.get('Roles', [])),
        Skills=safe_parse(SkillNode, raw_json.get('Skills', [])),
        Candidates=safe_parse(CandidateNode, raw_json.get('Candidates', [])),
        HAS_SUBDOMAIN=safe_parse(HasSubdomainRel, raw_json.get('HAS_SUBDOMAIN', [])),
        HAS_ROLE=safe_parse(HasRoleRel, raw_json.get('HAS_ROLE', [])),
        BELONGS_TO=safe_parse(BelongsToRel, raw_json.get('BELONGS_TO', [])),
        REQUIRES=safe_parse(RequiresRel, raw_json.get('REQUIRES', [])),
        HAS_SKILL=safe_parse(HasSkillRel, raw_json.get('HAS_SKILL', [])),
        SUITABLE_FOR=safe_parse(SuitableForRel, raw_json.get('SUITABLE_FOR', [])),
        RELATED_TO=safe_parse(RelatedToRel, raw_json.get('RELATED_TO', [])),
    )

# ══════════════════════════════════════════════════════════════
# PASS 3: REFERENCE VALIDATION & DEDUPLICATION
# ══════════════════════════════════════════════════════════════

FUZZY_THRESHOLD = 90

def find_canonical(name, norm_map, threshold=FUZZY_THRESHOLD):
    key = name.strip().lower()
    if key in norm_map: return norm_map[key], False
    best_score, best_match = 0, None
    for existing_key, original in norm_map.items():
        score = fuzz.ratio(key, existing_key)
        if score > best_score:
            best_score = score
            best_match = original
    if best_score >= threshold: return best_match, False
    return name.strip(), True

def pass3_validate(extraction, refs):
    nd, nsd, nr, ns = refs['norm_domains'], refs['norm_subdomains'], refs['norm_roles'], refs['norm_skills']
    
    # ── Enforce Disjoint Sets (Domain > SubDomain > Role > Skill) ──
    def is_known_in(*maps):
        def _check(name):
            key = name.strip().lower()
            return any(key in m for m in maps)
        return _check

    is_domain    = is_known_in(nd)
    is_subdomain = is_known_in(nd, nsd)
    is_role      = is_known_in(nd, nsd, nr)
    
    # Filter extraction lists before processing
    extraction.SubDomains = [sd for sd in extraction.SubDomains if not is_domain(sd.name)]
    extraction.Roles = [r for r in extraction.Roles if not is_subdomain(r.name)]
    extraction.Skills = [s for s in extraction.Skills if not is_role(s.name)]
    
    # Validate nodes
    for d in extraction.Domains:
        canonical, is_new = find_canonical(d.name, nd)
        d.name = canonical
        if is_new: nd[canonical.lower()] = canonical; refs['new_domains'].add(canonical)
    for sd in extraction.SubDomains:
        canonical, is_new = find_canonical(sd.name, nsd)
        sd.name = canonical
        if is_new: nsd[canonical.lower()] = canonical; refs['new_subdomains'].add(canonical)
    for r in extraction.Roles:
        canonical, is_new = find_canonical(r.name, nr)
        r.name = canonical
        if is_new: nr[canonical.lower()] = canonical; refs['new_roles'].add(canonical)
    for s in extraction.Skills:
        canonical, is_new = find_canonical(s.name, ns)
        if s.name.strip() != canonical:
            if s.aliases is None: s.aliases = []
            if s.name.strip() not in s.aliases: s.aliases.append(s.name.strip())
        s.name = canonical
        if is_new: ns[canonical.lower()] = canonical; refs['new_skills'].add(canonical)
    
    # Validate relationships
    for rel in extraction.HAS_SUBDOMAIN:
        rel.domain, _ = find_canonical(rel.domain, nd)
        rel.subdomain, _ = find_canonical(rel.subdomain, nsd)
    for rel in extraction.HAS_ROLE:
        rel.subdomain, _ = find_canonical(rel.subdomain, nsd)
        rel.role, _ = find_canonical(rel.role, nr)
    for rel in extraction.BELONGS_TO:
        rel.skill, _ = find_canonical(rel.skill, ns)
        rel.role, _ = find_canonical(rel.role, nr)
    for rel in extraction.REQUIRES:
        rel.role, _ = find_canonical(rel.role, nr)
        rel.skill, _ = find_canonical(rel.skill, ns)
    for rel in extraction.HAS_SKILL:
        rel.skill, _ = find_canonical(rel.skill, ns)
    for rel in extraction.SUITABLE_FOR:
        rel.role, _ = find_canonical(rel.role, nr)
    for rel in extraction.RELATED_TO:
        rel.skill_from, _ = find_canonical(rel.skill_from, ns)
        rel.skill_to, _ = find_canonical(rel.skill_to, ns)
    
    return extraction

# ══════════════════════════════════════════════════════════════
# NEO4J PUSH (with clean update support)
# ══════════════════════════════════════════════════════════════

def push_to_neo4j(extraction, is_update=False):
    with driver.session(database=DB) as session:
        # Clean old candidate-specific relationships on update
        if is_update:
            for c in extraction.Candidates:
                session.run("MATCH (c:Candidate {id: $cid})-[r:HAS_SKILL|SUITABLE_FOR]->() DELETE r", cid=c.id)
        
        # Nodes
        for d in extraction.Domains:
            session.run('MERGE (n:Domain {name: $name}) SET n.description = $desc, n.category = $cat', name=d.name, desc=d.description, cat=d.category)
        for sd in extraction.SubDomains:
            session.run('MERGE (n:SubDomain {name: $name}) SET n.core_focus = $focus', name=sd.name, focus=sd.core_focus)
        for r in extraction.Roles:
            session.run('MERGE (n:Role {name: $name}) SET n.description = $desc', name=r.name, desc=r.description)
        for s in extraction.Skills:
            session.run('MERGE (n:Skill {name: $name}) SET n.category = $cat, n.aliases = $aliases', name=s.name, cat=s.category, aliases=s.aliases or [])
        for c in extraction.Candidates:
            session.run('MERGE (n:Candidate {id: $id}) SET n.name = $name, n.experience = $exp', id=c.id, name=c.name, exp=c.experience)
        
        # Relationships
        for rel in extraction.HAS_SUBDOMAIN:
            session.run("MATCH (d:Domain {name: $dname}) MATCH (sd:SubDomain {name: $sdname}) MERGE (d)-[:HAS_SUBDOMAIN]->(sd)", dname=rel.domain, sdname=rel.subdomain)
        for rel in extraction.HAS_ROLE:
            session.run("MATCH (sd:SubDomain {name: $sdname}) MATCH (r:Role {name: $rname}) MERGE (sd)-[e:HAS_ROLE]->(r) SET e.priority = $p", sdname=rel.subdomain, rname=rel.role, p=rel.priority)
        for rel in extraction.BELONGS_TO:
            session.run("MATCH (s:Skill {name: $sname}) MATCH (r:Role {name: $rname}) MERGE (s)-[e:BELONGS_TO]->(r) SET e.is_core = $ic", sname=rel.skill, rname=rel.role, ic=rel.is_core)
        for rel in extraction.REQUIRES:
            session.run("MATCH (r:Role {name: $rname}) MATCH (s:Skill {name: $sname}) MERGE (r)-[e:REQUIRES]->(s) SET e.importance = $imp, e.min_level = $lvl", rname=rel.role, sname=rel.skill, imp=rel.importance, lvl=rel.min_level)
        for rel in extraction.HAS_SKILL:
            session.run("MATCH (c:Candidate {id: $cid}) MATCH (s:Skill {name: $sname}) MERGE (c)-[e:HAS_SKILL]->(s) SET e.proficiency_level = $prof, e.type = $stype", cid=rel.candidate, sname=rel.skill, prof=rel.proficiency_level, stype=rel.type)
        for rel in extraction.SUITABLE_FOR:
            session.run("MATCH (c:Candidate {id: $cid}) MATCH (r:Role {name: $rname}) MERGE (c)-[e:SUITABLE_FOR]->(r) SET e.level = $level", cid=rel.candidate, rname=rel.role, level=rel.level)
        for rel in extraction.RELATED_TO:
            session.run("MATCH (s1:Skill {name: $f}) MATCH (s2:Skill {name: $t}) MERGE (s1)-[e:RELATED_TO]->(s2) SET e.type = $rt, e.weight = $w",
                        f=rel.skill_from, t=rel.skill_to, rt=rel.type, w=rel.weight)

# ══════════════════════════════════════════════════════════════
# REFERENCE FILE UPDATER
# ══════════════════════════════════════════════════════════════

def update_reference_files(refs):
    updated = False
    
    # 1. Domains.md
    if refs['new_domains']:
        with open('domains.md', 'a', encoding='utf-8') as f:
            for d in sorted(refs['new_domains']):
                f.write(f'\n## {d}\n\n')
        print(f'    📝 Added {len(refs["new_domains"])} new domains to domains.md')
        updated = True
        
    # 2. Subdomains & Roles.md
    if refs['new_subdomains'] or refs['new_roles']:
        # Note: We append new subdomains/roles to a special section
        with open('subdomains_and _roles.md', 'a', encoding='utf-8') as f:
            if refs['new_subdomains']:
                f.write('\n## [NEWLY DISCOVERED SUBDOMAINS]\n')
                for sd in sorted(refs['new_subdomains']):
                    f.write(f'\n### {sd}\n')
            if refs['new_roles']:
                f.write('\n### [NEWLY DISCOVERED ROLES]\n')
                for r in sorted(refs['new_roles']):
                    f.write(f'* {r}\n')
        print(f'    📝 Updated subdomains_and _roles.md')
        updated = True
        
    # 3. Skills_and_role.md
    if refs['new_skills']:
        with open('skills_and_role.md', 'a', encoding='utf-8') as f:
            f.write('\n### [NEWLY DISCOVERED SKILLS]\n')
            for s in sorted(refs['new_skills']):
                f.write(f'* {s}\n')
        print(f'    📝 Added {len(refs["new_skills"])} new skills to skills_and_role.md')
        updated = True
        
    if not updated:
        print('    ℹ No new canonical entities discovered.')

# ══════════════════════════════════════════════════════════════
# CHECKPOINT (HASHING)
# ══════════════════════════════════════════════════════════════

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f: return json.load(f)
    return {}

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, 'w') as f: json.dump(cp, f, indent=2)

def get_row_hash(row):
    content = '|'.join(str(row.get(col, '')) for col in sorted(row.index))
    return hashlib.sha256(content.encode()).hexdigest()

# ══════════════════════════════════════════════════════════════
# MAIN SYNC ENGINE
# ══════════════════════════════════════════════════════════════

def sync_csv_to_graph():
    """Core sync logic — processes all changes between CSV and checkpoint."""
    print(f'\n{"="*60}')
    print(f'🔄 SYNC STARTED at {datetime.now().strftime("%H:%M:%S")}')
    print(f'{"="*60}')
    
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f'❌ Could not read CSV: {e}')
        return
    
    checkpoint = load_checkpoint()
    refs = load_all_references()
    system_prompt = build_system_prompt()
    
    stats = {'new': 0, 'updated': 0, 'skipped': 0, 'failed': 0, 'deleted': 0}
    current_ids = set()
    
    for index, row in df.iterrows():
        cid = str(row.get('id', ''))
        current_ids.add(cid)
        curr_hash = get_row_hash(row)
        
        # Skip unchanged
        if cid in checkpoint and checkpoint[cid]['hash'] == curr_hash:
            stats['skipped'] += 1
            continue
        
        is_update = cid in checkpoint
        label = "UPDATE" if is_update else "NEW"
        print(f'\n  [{label}] Candidate {cid}...')
        
        success = False
        paragraph = row_to_paragraph(row)
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.time()
                raw_json = pass1_extract(paragraph, system_prompt)
                if not raw_json:
                    print(f'    ⚠ Attempt {attempt}: LLM returned empty JSON')
                    if attempt < MAX_RETRIES:
                        delay = RETRY_DELAY * (2 ** (attempt - 1))
                        time.sleep(delay)
                    continue
                
                validated = pass3_validate(pass2_normalize(raw_json), refs)
                push_to_neo4j(validated, is_update=is_update)
                
                checkpoint[cid] = {'hash': curr_hash, 'ts': datetime.now().isoformat()}
                save_checkpoint(checkpoint)
                
                stats['updated' if is_update else 'new'] += 1
                print(f'    ✅ Attempt {attempt}: Done in {time.time()-t0:.1f}s')
                success = True
                break
            except Exception as e:
                print(f'    ⚠ Attempt {attempt}: Error — {e}')
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * (2 ** (attempt - 1))
                    time.sleep(delay)
        
        if not success:
            stats['failed'] += 1
            print(f'    ❌ Permanently failed after {MAX_RETRIES} attempts')
    
    # Deletion sweep
    deleted_ids = [cid for cid in list(checkpoint.keys()) if cid not in current_ids]
    if deleted_ids:
        print(f'\n  🗑 Removing {len(deleted_ids)} deleted candidates...')
        with driver.session(database=DB) as session:
            for cid in deleted_ids:
                session.run('MATCH (c:Candidate {id: $cid}) DETACH DELETE c', cid=cid)
                del checkpoint[cid]
                print(f'    Deleted {cid}')
        save_checkpoint(checkpoint)
        stats['deleted'] = len(deleted_ids)
    
    # Update .md files
    update_reference_files(refs)
    
    # Summary
    print(f'\n{"─"*60}')
    print(f'📊 SUMMARY: New={stats["new"]}, Updated={stats["updated"]}, '
          f'Skipped={stats["skipped"]}, Failed={stats["failed"]}, Deleted={stats["deleted"]}')
    print(f'{"─"*60}')

# ══════════════════════════════════════════════════════════════
# WATCHDOG FILE WATCHER
# ══════════════════════════════════════════════════════════════

class CSVChangeHandler(FileSystemEventHandler):
    """Watches for changes to profiles.csv and triggers sync."""
    
    def __init__(self):
        self.last_trigger = 0
    
    def on_modified(self, event):
        if not event.is_directory and os.path.basename(event.src_path) == CSV_FILE:
            now = time.time()
            # Debounce: wait DEBOUNCE_SECONDS after last save
            if now - self.last_trigger < DEBOUNCE_SECONDS:
                return
            self.last_trigger = now
            print(f'\n📁 Detected change in {CSV_FILE}!')
            time.sleep(1)  # Wait for file write to complete
            sync_csv_to_graph()

# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('+--------------------------------------------------------------+')
    print('|   KG WATCHER — Real-Time Knowledge Graph Sync                |')
    print('+--------------------------------------------------------------+')
    
    # Verify connections
    try:
        driver.verify_connectivity()
        print('✅ Neo4j connected')
    except Exception as e:
        print(f'❌ Neo4j failed: {e}'); exit(1)
    
    try:
        models = ollama.list()
        qwen_found = any('qwen2.5:7b' in m.model for m in models.models)
        print(f'✅ Ollama running. Qwen2.5:7b available: {qwen_found}')
    except Exception as e:
        print(f'❌ Ollama failed: {e}'); exit(1)
    
    # Initial sync
    print('\n🚀 Running initial sync...')
    sync_csv_to_graph()
    
    # Start watching
    print(f'\n👁 Watching {CSV_FILE} for changes... (Ctrl+C to stop)')
    observer = Observer()
    observer.schedule(CSVChangeHandler(), path='.', recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print('\n⏹ Watcher stopped.')
    
    observer.join()
    driver.close()
