#!/usr/bin/env python3
"""
Extract all summaries and bug reports from copilot.md

This script identifies and extracts:
- Session summaries (## 🎯 SESSION SUMMARY, ## 📊 Session Summary, etc.)
- Bug reports (✅ Bug #X, ❌ Bug #X, ❓ Bug #X)
- Commit summaries
- Major section conclusions

Output formats:
- JSON: Structured data with metadata (dates, commits, bugs)
- Markdown: Human-readable summary document
- Text: Plain text for quick review
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict


class SummaryExtractor:
    """Extract and structure summaries from copilot.md"""
    
    def __init__(self, input_file: str = "copilot.md"):
        self.input_file = Path(input_file)
        self.content = ""
        self.summaries = []
        self.bugs = defaultdict(list)
        
        # Patterns for different types of summaries
        self.patterns = {
            'session_summary': re.compile(
                r'^##\s+[🎯📊📋].*?(SESSION SUMMARY|RÉSUMÉ|Summary)',
                re.IGNORECASE | re.MULTILINE
            ),
            'bug_header': re.compile(
                r'^(✅|❌|❓|🐛)\s*\*\*Bug\s*#(\d+)[:\s]',
                re.MULTILINE
            ),
            'bug_inline': re.compile(
                r'Bug\s*#(\d+)[:\s]([^\n]+)',
                re.IGNORECASE
            ),
            'commit': re.compile(
                r'(Commit[:\s]+|commit\s+)([a-f0-9]{7,40})',
                re.IGNORECASE
            ),
            'kernel': re.compile(
                r'kernel[:\s]+([a-z]{4})',
                re.IGNORECASE
            ),
            'section_header': re.compile(
                r'^##\s+(.+?)$',
                re.MULTILINE
            )
        }
        
    def load_content(self):
        """Load copilot.md content"""
        print(f"📖 Loading {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.content = f.read()
        print(f"✅ Loaded {len(self.content):,} characters")
    
    def find_section_end(self, start_pos: int) -> int:
        """Find where a section ends (next ## heading or EOF)"""
        next_header = self.content.find('\n## ', start_pos + 1)
        return next_header if next_header != -1 else len(self.content)
    
    def extract_session_summaries(self) -> List[Dict[str, Any]]:
        """Extract all session summary sections"""
        summaries = []
        
        for match in self.patterns['session_summary'].finditer(self.content):
            start = match.start()
            end = self.find_section_end(start)
            
            section_content = self.content[start:end]
            header = match.group(0).strip()
            
            # Extract metadata from section
            bugs_found = self._extract_bugs_from_section(section_content)
            commits = self._extract_commits_from_section(section_content)
            kernels = self._extract_kernels_from_section(section_content)
            
            summary = {
                'type': 'session_summary',
                'header': header,
                'line': self.content[:start].count('\n') + 1,
                'content': section_content,
                'length': len(section_content),
                'bugs': bugs_found,
                'commits': commits,
                'kernels': kernels,
                'has_conclusion': any(word in section_content.lower() 
                                     for word in ['conclusion', 'résumé', 'summary'])
            }
            
            summaries.append(summary)
        
        print(f"📊 Found {len(summaries)} session summaries")
        return summaries
    
    def _extract_bugs_from_section(self, content: str) -> List[Dict[str, Any]]:
        """Extract bug information from a section"""
        bugs = []
        
        # Look for formatted bug entries (✅ **Bug #X: ...)
        for match in self.patterns['bug_header'].finditer(content):
            status = match.group(1)  # ✅, ❌, or ❓
            bug_num = match.group(2)
            
            # Extract the full line
            line_start = match.start()
            line_end = content.find('\n', line_start)
            if line_end == -1:
                line_end = len(content)
            
            bug_line = content[line_start:line_end].strip()
            
            # Determine status
            status_map = {
                '✅': 'resolved',
                '❌': 'discovered',
                '❓': 'investigating',
                '🐛': 'reported'
            }
            
            bugs.append({
                'number': int(bug_num),
                'status': status_map.get(status, 'unknown'),
                'description': bug_line,
                'raw': bug_line
            })
        
        # Also look for inline bug mentions
        for match in self.patterns['bug_inline'].finditer(content):
            bug_num = match.group(1)
            description = match.group(2).strip()
            
            # Avoid duplicates
            if not any(b['number'] == int(bug_num) for b in bugs):
                bugs.append({
                    'number': int(bug_num),
                    'status': 'mentioned',
                    'description': description,
                    'raw': match.group(0)
                })
        
        return sorted(bugs, key=lambda x: x['number'])
    
    def _extract_commits_from_section(self, content: str) -> List[str]:
        """Extract commit hashes from a section"""
        commits = []
        for match in self.patterns['commit'].finditer(content):
            commit_hash = match.group(2)
            if commit_hash not in commits:
                commits.append(commit_hash)
        return commits
    
    def _extract_kernels_from_section(self, content: str) -> List[str]:
        """Extract Kaggle kernel IDs from a section"""
        kernels = []
        for match in self.patterns['kernel'].finditer(content):
            kernel_id = match.group(1).lower()
            if kernel_id not in kernels and len(kernel_id) == 4:
                kernels.append(kernel_id)
        return kernels
    
    def extract_all_bugs(self) -> Dict[int, List[Dict[str, Any]]]:
        """Extract all bug mentions across the entire document"""
        bugs_by_number = defaultdict(list)
        
        # Search entire document for bug mentions
        for match in self.patterns['bug_header'].finditer(self.content):
            status = match.group(1)
            bug_num = int(match.group(2))
            
            # Get context (surrounding lines)
            line_start = max(0, match.start() - 200)
            line_end = min(len(self.content), match.end() + 500)
            context = self.content[line_start:line_end]
            
            # Extract the bug description line
            bug_line_start = match.start()
            bug_line_end = self.content.find('\n', bug_line_start)
            if bug_line_end == -1:
                bug_line_end = len(self.content)
            bug_description = self.content[bug_line_start:bug_line_end].strip()
            
            status_map = {
                '✅': 'resolved',
                '❌': 'discovered',
                '❓': 'investigating',
                '🐛': 'reported'
            }
            
            bugs_by_number[bug_num].append({
                'line': self.content[:match.start()].count('\n') + 1,
                'status': status_map.get(status, 'unknown'),
                'description': bug_description,
                'context': context.strip()
            })
        
        print(f"🐛 Found {len(bugs_by_number)} unique bugs with {sum(len(v) for v in bugs_by_number.values())} total mentions")
        return dict(bugs_by_number)
    
    def generate_json_output(self, output_file: str = "summaries_extracted.json"):
        """Generate JSON output with all extracted data"""
        data = {
            'metadata': {
                'source_file': str(self.input_file),
                'extraction_date': datetime.now().isoformat(),
                'total_size': len(self.content),
                'total_lines': self.content.count('\n')
            },
            'session_summaries': self.summaries,
            'bugs': self.bugs,
            'statistics': {
                'total_summaries': len(self.summaries),
                'total_bugs': len(self.bugs),
                'bugs_resolved': sum(1 for bug_entries in self.bugs.values() 
                                    if any(e['status'] == 'resolved' for e in bug_entries)),
                'bugs_discovered': sum(1 for bug_entries in self.bugs.values() 
                                      if any(e['status'] == 'discovered' for e in bug_entries))
            }
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON output saved to {output_path}")
        return data
    
    def generate_markdown_output(self, output_file: str = "summaries_extracted.md"):
        """Generate human-readable markdown summary"""
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 Extracted Summaries from copilot.md\n\n")
            f.write(f"**Extraction Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Source**: {self.input_file}\n\n")
            f.write("---\n\n")
            
            # Table of Contents
            f.write("## 📑 Table of Contents\n\n")
            f.write("- [Session Summaries](#session-summaries)\n")
            f.write("- [Bug Tracking](#bug-tracking)\n")
            f.write("- [Statistics](#statistics)\n\n")
            f.write("---\n\n")
            
            # Session Summaries
            f.write("## 🎯 Session Summaries\n\n")
            f.write(f"**Total Found**: {len(self.summaries)}\n\n")
            
            for i, summary in enumerate(self.summaries, 1):
                f.write(f"### Summary #{i}: {summary['header']}\n\n")
                f.write(f"- **Line**: {summary['line']}\n")
                f.write(f"- **Length**: {summary['length']:,} characters\n")
                
                if summary['bugs']:
                    bug_list = ', '.join(f"#{b['number']}" for b in summary['bugs'])
                    f.write(f"- **Bugs Mentioned**: {bug_list}\n")
                
                if summary['commits']:
                    f.write(f"- **Commits**: {', '.join(summary['commits'])}\n")
                
                if summary['kernels']:
                    f.write(f"- **Kernels**: {', '.join(summary['kernels'])}\n")
                
                f.write("\n**Content Preview**:\n\n")
                # First 500 characters of content
                preview = summary['content'][:500].strip()
                f.write(f"```\n{preview}...\n```\n\n")
                f.write("---\n\n")
            
            # Bug Tracking
            f.write("## 🐛 Bug Tracking\n\n")
            f.write(f"**Total Unique Bugs**: {len(self.bugs)}\n\n")
            
            for bug_num in sorted(self.bugs.keys()):
                entries = self.bugs[bug_num]
                f.write(f"### Bug #{bug_num}\n\n")
                f.write(f"**Total Mentions**: {len(entries)}\n\n")
                
                # Show status progression
                statuses = [e['status'] for e in entries]
                f.write(f"**Status Progression**: {' → '.join(statuses)}\n\n")
                
                # Show all entries
                for j, entry in enumerate(entries, 1):
                    f.write(f"#### Mention #{j} (Line {entry['line']})\n\n")
                    f.write(f"**Status**: {entry['status']}\n\n")
                    f.write(f"**Description**:\n```\n{entry['description']}\n```\n\n")
                
                f.write("---\n\n")
            
            # Statistics
            f.write("## 📈 Statistics\n\n")
            
            resolved = sum(1 for bug_entries in self.bugs.values() 
                          if any(e['status'] == 'resolved' for e in bug_entries))
            discovered = sum(1 for bug_entries in self.bugs.values() 
                            if any(e['status'] == 'discovered' for e in bug_entries))
            investigating = sum(1 for bug_entries in self.bugs.values() 
                               if any(e['status'] == 'investigating' for e in bug_entries))
            
            f.write(f"- **Total Session Summaries**: {len(self.summaries)}\n")
            f.write(f"- **Total Unique Bugs**: {len(self.bugs)}\n")
            f.write(f"- **Bugs Resolved**: {resolved} ✅\n")
            f.write(f"- **Bugs Discovered**: {discovered} ❌\n")
            f.write(f"- **Bugs Under Investigation**: {investigating} ❓\n")
            
            total_commits = len(set(c for s in self.summaries for c in s['commits']))
            total_kernels = len(set(k for s in self.summaries for k in s['kernels']))
            
            f.write(f"- **Total Commits Mentioned**: {total_commits}\n")
            f.write(f"- **Total Kaggle Kernels**: {total_kernels}\n")
        
        print(f"✅ Markdown output saved to {output_path}")
    
    def run(self):
        """Main extraction workflow"""
        print("\n🚀 Starting Summary Extraction\n")
        print("=" * 60)
        
        # Load content
        self.load_content()
        
        # Extract summaries
        print("\n📊 Extracting session summaries...")
        self.summaries = self.extract_session_summaries()
        
        # Extract all bugs
        print("\n🐛 Extracting bug mentions...")
        self.bugs = self.extract_all_bugs()
        
        # Generate outputs
        print("\n💾 Generating outputs...")
        self.generate_json_output()
        self.generate_markdown_output()
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("\n✨ EXTRACTION COMPLETE!\n")
        print(f"📊 Session Summaries: {len(self.summaries)}")
        print(f"🐛 Unique Bugs Found: {len(self.bugs)}")
        
        resolved = sum(1 for bug_entries in self.bugs.values() 
                      if any(e['status'] == 'resolved' for e in bug_entries))
        print(f"✅ Bugs Resolved: {resolved}")
        
        print("\n📁 Output files created:")
        print("   - summaries_extracted.json (structured data)")
        print("   - summaries_extracted.md (human-readable)")
        print("\n🎯 Done!\n")


if __name__ == "__main__":
    extractor = SummaryExtractor("copilot.md")
    extractor.run()
