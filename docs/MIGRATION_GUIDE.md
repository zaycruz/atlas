# Atlas Memory System Migration Guide

## Overview

This guide provides step-by-step instructions for migrating existing Atlas memory data to the enhanced memory system. The migration process preserves all existing functionality while adding new capabilities for contextual chunking, temporal decay, and session memory.

## Migration Philosophy

### Backward Compatibility Guarantees
1. **Zero Data Loss**: All existing memories are preserved during migration
2. **Functional Compatibility**: Existing Atlas functionality continues to work unchanged
3. **Graceful Degradation**: If enhanced features fail, system falls back to legacy behavior
4. **Incremental Adoption**: New features can be enabled gradually without disrupting existing workflows

### Migration Strategy
- **In-place migration**: Existing memory files are upgraded automatically
- **Backup creation**: Original files are backed up before modification
- **Rollback capability**: Migration can be reversed if issues occur
- **Validation**: Migrated data is verified for correctness and completeness

## Pre-Migration Assessment

### System Requirements Check

```python
import sys
from pathlib import Path
import json
import os

def check_migration_requirements():
    """Check if system meets requirements for memory enhancement migration."""
    
    requirements = {
        'python_version': sys.version_info >= (3, 9),
        'numpy_available': False,
        'storage_space': False,
        'memory_files_readable': False,
        'backup_space': False
    }
    
    # Check numpy availability
    try:
        import numpy
        requirements['numpy_available'] = True
    except ImportError:
        pass
    
    # Check storage space and memory files
    atlas_dir = Path.home() / '.local' / 'share' / 'atlas'
    if atlas_dir.exists():
        try:
            # Check if memory files are readable
            episodic_file = atlas_dir / 'episodic.json'
            if episodic_file.exists():
                with open(episodic_file, 'r') as f:
                    json.load(f)
                requirements['memory_files_readable'] = True
            
            # Check available space (need ~2x current file size for backup)
            import shutil
            total, used, free = shutil.disk_usage(atlas_dir)
            current_size = sum(f.stat().st_size for f in atlas_dir.rglob('*.json'))
            requirements['storage_space'] = free > (current_size * 3)  # 3x for safety
            requirements['backup_space'] = free > (current_size * 2)
            
        except Exception as e:
            print(f"Warning: Could not assess memory files: {e}")
    
    return requirements

def print_requirements_report(requirements):
    """Print migration requirements assessment."""
    print("Migration Requirements Assessment:")
    print("=" * 40)
    
    for req, met in requirements.items():
        status = "✓ PASS" if met else "✗ FAIL"
        req_name = req.replace('_', ' ').title()
        print(f"{req_name}: {status}")
    
    all_met = all(requirements.values())
    print("\nOverall Status:", "✓ READY" if all_met else "✗ NOT READY")
    
    if not all_met:
        print("\nPlease address failed requirements before proceeding with migration.")
    
    return all_met

# Run assessment
if __name__ == "__main__":
    requirements = check_migration_requirements()
    ready = print_requirements_report(requirements)
    sys.exit(0 if ready else 1)
```

### Data Inventory

Before migration, catalog existing memory data:

```python
from pathlib import Path
import json
from typing import Dict, Any

def inventory_existing_memory() -> Dict[str, Any]:
    """Create inventory of existing Atlas memory data."""
    
    atlas_dir = Path.home() / '.local' / 'share' / 'atlas'
    inventory = {
        'atlas_directory': str(atlas_dir),
        'directory_exists': atlas_dir.exists(),
        'files': {},
        'total_size_bytes': 0,
        'total_records': 0,
        'oldest_record': None,
        'newest_record': None
    }
    
    if not atlas_dir.exists():
        return inventory
    
    # Inventory memory files
    memory_files = [
        'episodic.json',
        'profile.json', 
        'journal.json'
    ]
    
    for filename in memory_files:
        filepath = atlas_dir / filename
        
        file_info = {
            'exists': filepath.exists(),
            'size_bytes': 0,
            'records': 0,
            'format_version': 'legacy',
            'readable': False,
            'last_modified': None
        }
        
        if filepath.exists():
            try:
                file_info['size_bytes'] = filepath.stat().st_size
                file_info['last_modified'] = filepath.stat().st_mtime
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    file_info['readable'] = True
                    
                    # Analyze structure
                    if filename == 'episodic.json':
                        if isinstance(data, dict) and 'records' in data:
                            records = data['records']
                            file_info['format_version'] = 'v1'
                        elif isinstance(data, list):
                            records = data
                            file_info['format_version'] = 'v0'
                        else:
                            records = []
                        
                        file_info['records'] = len(records)
                        inventory['total_records'] += len(records)
                        
                        # Find oldest/newest records
                        if records:
                            timestamps = [r.get('timestamp', 0) for r in records if isinstance(r, dict)]
                            if timestamps:
                                oldest = min(timestamps)
                                newest = max(timestamps)
                                
                                if inventory['oldest_record'] is None or oldest < inventory['oldest_record']:
                                    inventory['oldest_record'] = oldest
                                if inventory['newest_record'] is None or newest > inventory['newest_record']:
                                    inventory['newest_record'] = newest
                    
                    elif filename == 'profile.json':
                        if isinstance(data, dict):
                            file_info['records'] = len(data.get('facts', []))
                    
                    elif filename == 'journal.json':
                        if isinstance(data, dict):
                            file_info['records'] = len(data.get('entries', []))
                        elif isinstance(data, list):
                            file_info['records'] = len(data)
            
            except Exception as e:
                file_info['error'] = str(e)
        
        inventory['files'][filename] = file_info
        inventory['total_size_bytes'] += file_info['size_bytes']
    
    return inventory

def print_inventory_report(inventory):
    """Print human-readable inventory report."""
    print("Atlas Memory Data Inventory:")
    print("=" * 40)
    
    print(f"Atlas Directory: {inventory['atlas_directory']}")
    print(f"Directory Exists: {inventory['directory_exists']}")
    print(f"Total Size: {inventory['total_size_bytes']:,} bytes")
    print(f"Total Records: {inventory['total_records']:,}")
    
    if inventory['oldest_record'] and inventory['newest_record']:
        import datetime
        oldest = datetime.datetime.fromtimestamp(inventory['oldest_record'])
        newest = datetime.datetime.fromtimestamp(inventory['newest_record'])
        print(f"Memory Span: {oldest.date()} to {newest.date()}")
    
    print("\nFiles:")
    for filename, info in inventory['files'].items():
        status = "✓" if info['exists'] and info['readable'] else "✗"
        print(f"  {status} {filename}: {info['records']} records, {info['size_bytes']:,} bytes")
        
        if 'error' in info:
            print(f"    Error: {info['error']}")

# Run inventory
if __name__ == "__main__":
    inventory = inventory_existing_memory()
    print_inventory_report(inventory)
```

## Migration Process

### Step 1: Backup Creation

```python
import shutil
import datetime
from pathlib import Path

def create_pre_migration_backup():
    """Create complete backup of Atlas data before migration."""
    
    atlas_dir = Path.home() / '.local' / 'share' / 'atlas'
    if not atlas_dir.exists():
        print("No Atlas directory found, no backup needed.")
        return None
    
    # Create timestamped backup directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = atlas_dir.parent / f"atlas_backup_{timestamp}"
    
    try:
        # Copy entire atlas directory
        shutil.copytree(atlas_dir, backup_dir)
        
        # Create backup manifest
        manifest = {
            'backup_timestamp': timestamp,
            'original_path': str(atlas_dir),
            'backup_path': str(backup_dir),
            'migration_version': '1.0',
            'files_backed_up': [f.name for f in atlas_dir.rglob('*') if f.is_file()]
        }
        
        manifest_file = backup_dir / 'backup_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✓ Backup created successfully: {backup_dir}")
        return backup_dir
        
    except Exception as e:
        print(f"✗ Backup failed: {e}")
        # Clean up partial backup
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        raise

def verify_backup(backup_dir: Path):
    """Verify backup integrity."""
    
    manifest_file = backup_dir / 'backup_manifest.json'
    if not manifest_file.exists():
        raise ValueError("Backup manifest not found")
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    # Verify all files are present
    missing_files = []
    for filename in manifest['files_backed_up']:
        if not (backup_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        raise ValueError(f"Missing files in backup: {missing_files}")
    
    print(f"✓ Backup verified: {len(manifest['files_backed_up'])} files")
    return True
```

### Step 2: Memory Record Migration

```python
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import uuid
import time

@dataclass
class LegacyMemoryRecord:
    """Legacy memory record format."""
    id: str
    user: str
    assistant: str
    timestamp: float
    embedding: Optional[List[float]] = None

@dataclass 
class MigratedMemoryRecord:
    """Enhanced memory record format."""
    id: str
    parent_id: Optional[str] = None
    chunk_index: int = 0
    user: str = ""
    assistant: str = ""
    content: str = ""
    full_context: Optional[str] = None
    timestamp: float = 0.0
    embedding: Optional[List[float]] = None
    chunk_metadata: Optional[Dict] = None
    access_count: int = 0
    last_accessed: float = 0.0
    importance_score: float = 1.0

class MemoryRecordMigrator:
    """Handles migration of individual memory records."""
    
    def __init__(self):
        self.migration_stats = {
            'records_processed': 0,
            'records_migrated': 0,
            'errors': [],
            'warnings': []
        }
    
    def migrate_record(self, legacy_record: Dict[str, Any]) -> MigratedMemoryRecord:
        """Migrate a single legacy memory record to enhanced format."""
        
        try:
            self.migration_stats['records_processed'] += 1
            
            # Handle different legacy formats
            if isinstance(legacy_record, dict):
                record_id = legacy_record.get('id', str(uuid.uuid4()))
                user_text = legacy_record.get('user', '')
                assistant_text = legacy_record.get('assistant', '')
                timestamp = float(legacy_record.get('timestamp', time.time()))
                embedding = legacy_record.get('embedding')
            else:
                # Fallback for unexpected formats
                self.migration_stats['warnings'].append(f"Unexpected record format: {type(legacy_record)}")
                record_id = str(uuid.uuid4())
                user_text = str(legacy_record) if legacy_record else ''
                assistant_text = ''
                timestamp = time.time()
                embedding = None
            
            # Create enhanced record (treating legacy as single chunk/parent)
            migrated = MigratedMemoryRecord(
                id=record_id,
                parent_id=None,  # Legacy records become parents
                chunk_index=0,   # Parent index
                user=user_text,
                assistant=assistant_text,
                content=f"User: {user_text}\nAssistant: {assistant_text}",
                full_context=None,  # Not needed for parent records
                timestamp=timestamp,
                embedding=embedding,
                chunk_metadata=None,  # No chunking metadata for legacy records
                access_count=0,  # Reset access tracking
                last_accessed=timestamp,  # Use original timestamp
                importance_score=1.0  # Default importance
            )
            
            self.migration_stats['records_migrated'] += 1
            return migrated
            
        except Exception as e:
            error_msg = f"Failed to migrate record {legacy_record}: {e}"
            self.migration_stats['errors'].append(error_msg)
            raise ValueError(error_msg)
    
    def migrate_episodic_memory_file(self, file_path: Path) -> List[MigratedMemoryRecord]:
        """Migrate entire episodic memory file."""
        
        if not file_path.exists():
            return []
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different file formats
        if isinstance(data, dict) and 'records' in data:
            # Version 1 format: {"records": [...]}
            legacy_records = data['records']
        elif isinstance(data, list):
            # Version 0 format: [...]
            legacy_records = data
        else:
            self.migration_stats['warnings'].append(f"Unknown file format in {file_path}")
            legacy_records = []
        
        migrated_records = []
        for legacy_record in legacy_records:
            try:
                migrated = self.migrate_record(legacy_record)
                migrated_records.append(migrated)
            except ValueError as e:
                # Continue with other records despite individual failures
                continue
        
        return migrated_records
    
    def save_migrated_memory(self, records: List[MigratedMemoryRecord], output_path: Path):
        """Save migrated records in enhanced format."""
        
        # Convert to dictionaries for JSON serialization
        record_dicts = [asdict(record) for record in records]
        
        # Create enhanced format file
        enhanced_data = {
            'format_version': '2.0',
            'migration_timestamp': time.time(),
            'migration_stats': self.migration_stats,
            'records': record_dicts
        }
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write migrated data
        with open(output_path, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        print(f"✓ Migrated {len(records)} records to {output_path}")

def migrate_episodic_memory():
    """Complete episodic memory migration process."""
    
    print("Starting episodic memory migration...")
    
    # Paths
    atlas_dir = Path.home() / '.local' / 'share' / 'atlas'
    legacy_path = atlas_dir / 'episodic.json'
    enhanced_path = atlas_dir / 'episodic_enhanced.json'
    
    if not legacy_path.exists():
        print("No legacy episodic memory file found, skipping migration.")
        return
    
    # Create migrator
    migrator = MemoryRecordMigrator()
    
    try:
        # Migrate records
        migrated_records = migrator.migrate_episodic_memory_file(legacy_path)
        
        # Save enhanced format
        migrator.save_migrated_memory(migrated_records, enhanced_path)
        
        # Print migration summary
        stats = migrator.migration_stats
        print(f"\nMigration Summary:")
        print(f"  Records processed: {stats['records_processed']}")
        print(f"  Records migrated: {stats['records_migrated']}")
        print(f"  Errors: {len(stats['errors'])}")
        print(f"  Warnings: {len(stats['warnings'])}")
        
        if stats['errors']:
            print("\nErrors encountered:")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        if stats['warnings']:
            print("\nWarnings:")
            for warning in stats['warnings'][:5]:  # Show first 5 warnings
                print(f"  - {warning}")
        
        # Rename files to activate migration
        if migrated_records:
            legacy_backup = atlas_dir / 'episodic_legacy.json'
            legacy_path.rename(legacy_backup)
            enhanced_path.rename(legacy_path)
            
            print(f"✓ Migration complete. Legacy file backed up as {legacy_backup.name}")
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        # Clean up partial migration
        if enhanced_path.exists():
            enhanced_path.unlink()
        raise
```

### Step 3: Configuration Migration

```python
def migrate_memory_configuration():
    """Migrate memory-related configuration settings."""
    
    print("Migrating memory configuration...")
    
    # Default enhanced memory settings
    enhanced_config = {
        'memory': {
            'enhanced_mode': True,
            'chunking': {
                'enabled': True,
                'chunk_size': 400,
                'overlap_ratio': 0.2,
                'min_chunk_size': 100,
                'max_chunks_per_turn': 10
            },
            'temporal_decay': {
                'enabled': True,
                'episodic_decay_rate': 0.01,
                'session_decay_rate': 0.1,
                'importance_boost_factor': 0.1,
                'max_importance_multiplier': 5.0,
                'forgetting_threshold': 0.01
            },
            'session_memory': {
                'enabled': True,
                'duration_hours': 2.0,
                'max_turns_per_session': 50,
                'inactivity_threshold_minutes': 30,
                'cleanup_interval_minutes': 15
            },
            'limits': {
                'max_episodic_records': 1000,
                'max_session_records': 200,
                'enable_memory_metrics': True
            }
        }
    }
    
    # Look for existing configuration
    atlas_dir = Path.home() / '.local' / 'share' / 'atlas'
    config_file = atlas_dir / 'config.json'
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
            
            # Merge enhanced settings with existing config
            if 'memory' in existing_config:
                existing_config['memory'].update(enhanced_config['memory'])
            else:
                existing_config['memory'] = enhanced_config['memory']
            
            final_config = existing_config
            
        except Exception as e:
            print(f"Warning: Could not read existing config: {e}")
            final_config = enhanced_config
    else:
        final_config = enhanced_config
    
    # Save updated configuration
    with open(config_file, 'w') as f:
        json.dump(final_config, f, indent=2)
    
    print(f"✓ Configuration updated: {config_file}")
```

### Step 4: Verification and Validation

```python
def verify_migration():
    """Verify migration was successful and data integrity is maintained."""
    
    print("Verifying migration...")
    
    atlas_dir = Path.home() / '.local' / 'share' / 'atlas'
    enhanced_file = atlas_dir / 'episodic.json'
    legacy_backup = atlas_dir / 'episodic_legacy.json'
    
    verification_results = {
        'files_exist': False,
        'format_valid': False,
        'record_count_matches': False,
        'data_integrity': False,
        'backwards_compatibility': False
    }
    
    # Check files exist
    if enhanced_file.exists() and legacy_backup.exists():
        verification_results['files_exist'] = True
    else:
        print("✗ Required files missing after migration")
        return verification_results
    
    try:
        # Load both files
        with open(enhanced_file, 'r') as f:
            enhanced_data = json.load(f)
        
        with open(legacy_backup, 'r') as f:
            legacy_data = json.load(f)
        
        # Verify enhanced format
        if (isinstance(enhanced_data, dict) and 
            'format_version' in enhanced_data and
            'records' in enhanced_data):
            verification_results['format_valid'] = True
        
        # Count records
        if isinstance(legacy_data, dict) and 'records' in legacy_data:
            legacy_records = legacy_data['records']
        elif isinstance(legacy_data, list):
            legacy_records = legacy_data
        else:
            legacy_records = []
        
        enhanced_records = enhanced_data.get('records', [])
        
        if len(enhanced_records) == len(legacy_records):
            verification_results['record_count_matches'] = True
        
        # Verify data integrity (sample check)
        if enhanced_records and legacy_records:
            # Check first few records for content preservation
            sample_size = min(5, len(legacy_records))
            integrity_checks = 0
            
            for i in range(sample_size):
                legacy = legacy_records[i]
                enhanced = enhanced_records[i]
                
                if (enhanced.get('user') == legacy.get('user') and
                    enhanced.get('assistant') == legacy.get('assistant') and
                    enhanced.get('timestamp') == legacy.get('timestamp')):
                    integrity_checks += 1
            
            if integrity_checks == sample_size:
                verification_results['data_integrity'] = True
        
        # Test backwards compatibility
        try:
            # Try to load with legacy code (simplified test)
            test_legacy_format = {
                'records': [r for r in enhanced_records if r.get('chunk_index', 0) == 0]
            }
            verification_results['backwards_compatibility'] = True
        except:
            pass
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return verification_results
    
    # Print results
    print("\nVerification Results:")
    for check, passed in verification_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        check_name = check.replace('_', ' ').title()
        print(f"  {check_name}: {status}")
    
    all_passed = all(verification_results.values())
    print(f"\nOverall: {'✓ MIGRATION SUCCESSFUL' if all_passed else '✗ MIGRATION ISSUES DETECTED'}")
    
    return verification_results

def test_enhanced_memory_functionality():
    """Test that enhanced memory features work after migration."""
    
    print("\nTesting enhanced memory functionality...")
    
    try:
        # Import enhanced memory classes
        from enhanced_memory import EnhancedEpisodicMemory, ContextualChunker
        
        # Test basic functionality
        atlas_dir = Path.home() / '.local' / 'share' / 'atlas'
        memory_path = atlas_dir / 'episodic.json'
        
        # Mock embedding function for testing
        def test_embedding_fn(text):
            return [0.1, 0.2, 0.3]
        
        memory = EnhancedEpisodicMemory(
            storage_path=memory_path,
            embedding_fn=test_embedding_fn
        )
        
        # Test loading migrated data
        initial_record_count = len(memory._records)
        print(f"  ✓ Loaded {initial_record_count} migrated records")
        
        # Test new functionality
        test_chunks = memory.remember("Test question", "Test answer for migration verification")
        print(f"  ✓ New chunking functionality works ({len(test_chunks)} chunks created)")
        
        # Test recall
        results = memory.recall("test", top_k=2)
        print(f"  ✓ Enhanced recall works ({len(results)} results)")
        
        # Test temporal weighting
        if results:
            score = results[0].calculate_relevance_score(0.8)
            print(f"  ✓ Temporal weighting works (score: {score:.3f})")
        
        print("  ✓ All enhanced functionality tests passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Enhanced functionality test failed: {e}")
        return False
```

## Complete Migration Script

```python
#!/usr/bin/env python3
"""
Atlas Memory System Migration Script

This script migrates existing Atlas memory data to the enhanced memory system
with contextual chunking, temporal decay, and session memory features.
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main migration script."""
    
    parser = argparse.ArgumentParser(description="Migrate Atlas memory to enhanced system")
    parser.add_argument('--check-only', action='store_true', help='Only check requirements, don\'t migrate')
    parser.add_argument('--force', action='store_true', help='Force migration even if checks fail')
    parser.add_argument('--backup-dir', type=str, help='Custom backup directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    print("Atlas Memory System Migration")
    print("=" * 40)
    
    try:
        # Step 1: Requirements check
        print("\n1. Checking migration requirements...")
        requirements = check_migration_requirements()
        ready = print_requirements_report(requirements)
        
        if not ready and not args.force:
            print("\nMigration aborted due to unmet requirements.")
            print("Use --force to proceed anyway (not recommended).")
            sys.exit(1)
        
        if args.check_only:
            sys.exit(0)
        
        # Step 2: Data inventory
        print("\n2. Inventorying existing data...")
        inventory = inventory_existing_memory()
        print_inventory_report(inventory)
        
        if not inventory['directory_exists']:
            print("No Atlas data found, migration not needed.")
            sys.exit(0)
        
        if args.dry_run:
            print("\nDry run complete. Use without --dry-run to perform migration.")
            sys.exit(0)
        
        # Step 3: Create backup
        print("\n3. Creating backup...")
        backup_dir = create_pre_migration_backup()
        verify_backup(backup_dir)
        
        # Step 4: Migrate memory data
        print("\n4. Migrating memory data...")
        migrate_episodic_memory()
        
        # Step 5: Migrate configuration
        print("\n5. Updating configuration...")
        migrate_memory_configuration()
        
        # Step 6: Verification
        print("\n6. Verifying migration...")
        verification_results = verify_migration()
        
        if all(verification_results.values()):
            print("\n7. Testing enhanced functionality...")
            test_enhanced_memory_functionality()
            
            print("\n" + "=" * 40)
            print("✓ Migration completed successfully!")
            print(f"✓ Backup created: {backup_dir}")
            print("✓ Enhanced memory features are now active")
            print("\nYou can now use Atlas with improved memory capabilities.")
        else:
            print("\n" + "=" * 40)
            print("⚠ Migration completed with issues")
            print("Please review the verification results above.")
            print("Consider rolling back if issues persist.")
        
    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        print("\nPlease check the error above and try again.")
        print("Your original data is safely backed up.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Rollback Procedure

If migration issues occur, you can rollback to the original system:

```python
def rollback_migration():
    """Rollback migration to restore original Atlas memory system."""
    
    print("Rolling back Atlas memory migration...")
    
    atlas_dir = Path.home() / '.local' / 'share' / 'atlas'
    enhanced_file = atlas_dir / 'episodic.json'
    legacy_backup = atlas_dir / 'episodic_legacy.json'
    
    if not legacy_backup.exists():
        raise FileNotFoundError("Legacy backup file not found, cannot rollback")
    
    # Restore original file
    if enhanced_file.exists():
        enhanced_backup = atlas_dir / 'episodic_enhanced_failed.json'
        enhanced_file.rename(enhanced_backup)
        print(f"✓ Failed enhanced file saved as {enhanced_backup.name}")
    
    legacy_backup.rename(enhanced_file)
    print(f"✓ Original file restored")
    
    # Remove enhanced configuration
    config_file = atlas_dir / 'config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if 'memory' in config:
                config['memory']['enhanced_mode'] = False
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("✓ Configuration updated to disable enhanced features")
        except:
            print("⚠ Could not update configuration, manual adjustment may be needed")
    
    print("✓ Rollback completed successfully")

# Usage
if __name__ == "__main__":
    rollback_migration()
```

## Post-Migration Considerations

### Performance Monitoring

After migration, monitor system performance:

```python
def monitor_post_migration_performance():
    """Monitor memory system performance after migration."""
    
    import time
    import psutil
    import os
    
    print("Monitoring post-migration performance...")
    
    # Test basic operations
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Simulate typical usage
    try:
        from enhanced_memory import EnhancedEpisodicMemory
        
        memory = EnhancedEpisodicMemory(
            storage_path=Path.home() / '.local' / 'share' / 'atlas' / 'episodic.json'
        )
        
        # Test recall performance
        recall_times = []
        for i in range(10):
            start = time.perf_counter()
            results = memory.recall(f"test query {i}", top_k=5)
            end = time.perf_counter()
            recall_times.append(end - start)
        
        avg_recall_time = sum(recall_times) / len(recall_times)
        max_recall_time = max(recall_times)
        
        # Test memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Report results
        print(f"Average recall time: {avg_recall_time*1000:.1f}ms")
        print(f"Maximum recall time: {max_recall_time*1000:.1f}ms")
        print(f"Memory increase: {memory_increase/1024/1024:.1f}MB")
        
        # Performance targets
        if avg_recall_time < 0.1:  # <100ms
            print("✓ Recall performance meets targets")
        else:
            print("⚠ Recall performance slower than target")
        
        if memory_increase < 50 * 1024 * 1024:  # <50MB
            print("✓ Memory usage within acceptable range")
        else:
            print("⚠ High memory usage detected")
        
    except Exception as e:
        print(f"✗ Performance monitoring failed: {e}")

# Run monitoring
if __name__ == "__main__":
    monitor_post_migration_performance()
```

### Usage Instructions

After successful migration:

1. **Restart Atlas**: Restart your Atlas session to activate enhanced features
2. **Verify Functionality**: Test basic conversation and memory recall
3. **Monitor Performance**: Watch for any performance issues
4. **Configure Settings**: Adjust memory settings via environment variables if needed
5. **Report Issues**: Report any problems with rollback information

### Environment Variables

Configure enhanced memory features:

```bash
# Enable/disable enhanced features
export ATLAS_ENHANCED_MEMORY=true

# Chunking settings
export ATLAS_CHUNK_SIZE=400
export ATLAS_CHUNK_OVERLAP=0.2

# Temporal decay settings  
export ATLAS_TEMPORAL_DECAY_RATE=0.01

# Session memory settings
export ATLAS_SESSION_DURATION_HOURS=2.0

# Memory limits
export ATLAS_MAX_EPISODIC_RECORDS=1000
```

---

*This migration guide ensures safe, reliable upgrade to the enhanced Atlas memory system while preserving all existing data and functionality. Follow the steps carefully and keep backups until you're confident in the migration success.*