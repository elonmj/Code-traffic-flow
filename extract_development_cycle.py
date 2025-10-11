#!/usr/bin/env python3
"""
Extract Development Cycle Patterns from copilot.md

This advanced script analyzes the development workflow in copilot.md to extract:
- Workflow phases (context, research, analysis, implementation, test, debug, validation)
- Tool call sequences (read_file → grep_search → replace_string → run_terminal)
- Iteration patterns (error → fix → test cycles)
- Decision points and reasoning
- Success/failure transitions
- Meta-cognitive patterns

The goal is to formalize the development cycle logic for reproducibility.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict


@dataclass
class WorkflowPhase:
    """Represents a detected workflow phase"""
    phase_type: str  # 'context', 'research', 'analysis', 'implementation', 'test', 'debug', 'validation'
    start_line: int
    end_line: int
    duration_lines: int
    tools_used: List[str]
    actions: List[str]
    outcome: str  # 'success', 'failure', 'partial', 'blocked'
    content_preview: str


@dataclass
class IterationCycle:
    """Represents an iteration cycle (attempt to solve a problem)"""
    cycle_id: int
    start_line: int
    end_line: int
    phases: List[WorkflowPhase]
    iterations_count: int
    final_outcome: str
    key_decisions: List[str]
    tools_sequence: List[str]


@dataclass
class ToolSequence:
    """Represents a common sequence of tool calls"""
    tools: List[str]
    frequency: int
    context: str  # What was being attempted
    typical_outcome: str


class DevelopmentCycleExtractor:
    """Extract and analyze development workflow patterns"""
    
    def __init__(self, input_file: str = "copilot.md"):
        self.input_file = Path(input_file)
        self.content = ""
        self.lines = []
        
        # Workflow analysis results
        self.phases = []
        self.cycles = []
        self.tool_sequences = []
        self.decision_points = []
        
        # Patterns for workflow detection
        self.patterns = {
            # Tool calls
            'read_file': re.compile(r'Read.*?lines?\s+\d+\s+to\s+\d+', re.IGNORECASE),
            'grep_search': re.compile(r'Searched text for', re.IGNORECASE),
            'semantic_search': re.compile(r'semantic[_\s]search', re.IGNORECASE),
            'replace_string': re.compile(r'Using "Replace String in File"', re.IGNORECASE),
            'create_file': re.compile(r'(create|creat(e|ing))\s+file', re.IGNORECASE),
            'run_terminal': re.compile(r'Ran terminal command:', re.IGNORECASE),
            'get_errors': re.compile(r'get[_\s]errors|check.*errors', re.IGNORECASE),
            
            # Workflow phases
            'context_gathering': re.compile(
                r'(let me (check|read|examine|look at)|I need to (check|read|see)|'
                r'First.*?(understand|check|read)|Now let me check)',
                re.IGNORECASE
            ),
            'research': re.compile(
                r'(search|research|look up|documentation|docs|investigate|'
                r'I need to understand)',
                re.IGNORECASE
            ),
            'analysis': re.compile(
                r'(the (issue|problem|error) is|I see|analysis|analyzing|'
                r'the root cause|identified)',
                re.IGNORECASE
            ),
            'implementation': re.compile(
                r'(I\'?ll (implement|add|create|write)|let me (fix|update|modify)|'
                r'implementing|creating)',
                re.IGNORECASE
            ),
            'testing': re.compile(
                r'(test|testing|verify|validat(e|ing)|run|running|check.*work)',
                re.IGNORECASE
            ),
            'debugging': re.compile(
                r'(debug|error|fix|issue|problem|fail(ed|ure)?|troubleshoot)',
                re.IGNORECASE
            ),
            
            # Outcomes
            'success': re.compile(
                r'(✅|✓|success|excellent|perfect|completed|works?|fixed)',
                re.IGNORECASE
            ),
            'failure': re.compile(
                r'(❌|✗|error|failed|failure|didn\'?t work|issue|problem)',
                re.IGNORECASE
            ),
            'partial': re.compile(
                r'(partial|almost|but|however|still (need|have))',
                re.IGNORECASE
            ),
            
            # Decision points
            'decision': re.compile(
                r'(I (will|\'ll|should|need to)|let me|the (plan|strategy|approach) is|'
                r'decision|choose|option)',
                re.IGNORECASE
            ),
            
            # Meta-cognitive markers
            'understanding': re.compile(
                r'(I understand|I see|now I (know|understand)|this means)',
                re.IGNORECASE
            ),
            'uncertainty': re.compile(
                r'(not sure|unsure|unclear|confus(ed|ing)|strange|unexpected)',
                re.IGNORECASE
            ),
            
            # Iteration markers
            'retry': re.compile(
                r'(re-?(run|try|test|attempt)|try again|let\'?s try|another attempt)',
                re.IGNORECASE
            ),
            'iteration': re.compile(
                r'(iteration|attempt #?\d+|cycle \d+|pass \d+)',
                re.IGNORECASE
            ),
        }
    
    def load_content(self):
        """Load copilot.md content"""
        print(f"📖 Loading {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.content = f.read()
            self.lines = self.content.split('\n')
        print(f"✅ Loaded {len(self.content):,} characters ({len(self.lines):,} lines)")
    
    def detect_tool_calls(self) -> List[Tuple[int, str, str]]:
        """Detect all tool calls with their line numbers and context"""
        tool_calls = []
        
        for tool_name, pattern in self.patterns.items():
            if tool_name in ['read_file', 'grep_search', 'semantic_search', 
                            'replace_string', 'create_file', 'run_terminal', 'get_errors']:
                for match in pattern.finditer(self.content):
                    line_num = self.content[:match.start()].count('\n') + 1
                    
                    # Get context (surrounding lines)
                    context_start = max(0, match.start() - 200)
                    context_end = min(len(self.content), match.end() + 100)
                    context = self.content[context_start:context_end]
                    
                    tool_calls.append((line_num, tool_name, context))
        
        # Sort by line number
        tool_calls.sort(key=lambda x: x[0])
        print(f"🔧 Found {len(tool_calls)} tool calls")
        return tool_calls
    
    def detect_workflow_phases(self) -> List[WorkflowPhase]:
        """Detect workflow phases based on actions and context"""
        phases = []
        current_phase = None
        phase_start = 0
        phase_tools = []
        phase_actions = []
        
        for i, line in enumerate(self.lines):
            # Detect phase transitions
            phase_type = None
            action = None
            
            for phase_name in ['context_gathering', 'research', 'analysis', 
                               'implementation', 'testing', 'debugging']:
                if self.patterns[phase_name].search(line):
                    phase_type = phase_name
                    action = line.strip()
                    break
            
            # Detect tool usage in this line
            for tool_name in ['read_file', 'grep_search', 'replace_string', 
                            'run_terminal', 'create_file']:
                if self.patterns[tool_name].search(line):
                    phase_tools.append(tool_name)
            
            # If we detected a new phase type
            if phase_type and (current_phase is None or phase_type != current_phase):
                # Save previous phase if exists
                if current_phase:
                    outcome = self._detect_outcome(i - 1, phase_start)
                    preview = '\n'.join(self.lines[phase_start:min(i, phase_start + 5)])
                    
                    phases.append(WorkflowPhase(
                        phase_type=current_phase,
                        start_line=phase_start + 1,
                        end_line=i,
                        duration_lines=i - phase_start,
                        tools_used=list(set(phase_tools)),
                        actions=phase_actions[:5],  # Keep first 5 actions
                        outcome=outcome,
                        content_preview=preview
                    ))
                
                # Start new phase
                current_phase = phase_type
                phase_start = i
                phase_tools = []
                phase_actions = []
            
            if action:
                phase_actions.append(action)
        
        # Save final phase
        if current_phase:
            outcome = self._detect_outcome(len(self.lines) - 1, phase_start)
            preview = '\n'.join(self.lines[phase_start:min(len(self.lines), phase_start + 5)])
            
            phases.append(WorkflowPhase(
                phase_type=current_phase,
                start_line=phase_start + 1,
                end_line=len(self.lines),
                duration_lines=len(self.lines) - phase_start,
                tools_used=list(set(phase_tools)),
                actions=phase_actions[:5],
                outcome=outcome,
                content_preview=preview
            ))
        
        print(f"🔄 Detected {len(phases)} workflow phases")
        return phases
    
    def _detect_outcome(self, end_line: int, start_line: int) -> str:
        """Detect the outcome of a phase"""
        section = '\n'.join(self.lines[start_line:end_line + 1])
        
        if self.patterns['success'].search(section):
            return 'success'
        elif self.patterns['failure'].search(section):
            return 'failure'
        elif self.patterns['partial'].search(section):
            return 'partial'
        else:
            return 'unknown'
    
    def detect_iteration_cycles(self, tool_calls: List[Tuple[int, str, str]]) -> List[IterationCycle]:
        """Detect iteration cycles (repeated attempts to solve a problem)"""
        cycles = []
        current_cycle_tools = []
        cycle_start_line = 0
        iteration_count = 0
        
        # Group tool calls into cycles
        for i, (line_num, tool, context) in enumerate(tool_calls):
            current_cycle_tools.append(tool)
            
            # Detect cycle completion (success or giving up)
            is_success = self.patterns['success'].search(context)
            is_failure = self.patterns['failure'].search(context)
            is_retry = self.patterns['retry'].search(context)
            
            # Check if next tool is a retry (indicates new iteration)
            if is_retry or (i < len(tool_calls) - 1 and tool_calls[i + 1][1] == tool):
                iteration_count += 1
            
            # Cycle ends on success or major failure
            if is_success or (is_failure and not is_retry):
                if current_cycle_tools:
                    outcome = 'success' if is_success else 'failure'
                    
                    # Extract phases for this cycle
                    cycle_phases = [p for p in self.phases 
                                   if cycle_start_line <= p.start_line <= line_num]
                    
                    # Extract decisions
                    decisions = self._extract_decisions(cycle_start_line, line_num)
                    
                    cycles.append(IterationCycle(
                        cycle_id=len(cycles) + 1,
                        start_line=cycle_start_line,
                        end_line=line_num,
                        phases=cycle_phases,
                        iterations_count=max(1, iteration_count),
                        final_outcome=outcome,
                        key_decisions=decisions,
                        tools_sequence=current_cycle_tools.copy()
                    ))
                    
                    # Reset for next cycle
                    current_cycle_tools = []
                    cycle_start_line = line_num + 1
                    iteration_count = 0
        
        print(f"🔁 Detected {len(cycles)} iteration cycles")
        return cycles
    
    def _extract_decisions(self, start_line: int, end_line: int) -> List[str]:
        """Extract decision points from a line range"""
        decisions = []
        section = '\n'.join(self.lines[start_line:end_line + 1])
        
        for match in self.patterns['decision'].finditer(section):
            # Get the full line
            line_start = match.start()
            line_end = section.find('\n', line_start)
            if line_end == -1:
                line_end = len(section)
            
            decision_text = section[line_start:line_end].strip()
            if len(decision_text) > 20 and len(decision_text) < 200:
                decisions.append(decision_text)
        
        return decisions[:5]  # Return up to 5 decisions
    
    def analyze_tool_sequences(self, tool_calls: List[Tuple[int, str, str]]) -> List[ToolSequence]:
        """Analyze common tool call sequences"""
        sequences = defaultdict(lambda: {'count': 0, 'contexts': [], 'outcomes': []})
        
        # Look for sequences of 2-4 tools
        for seq_len in [2, 3, 4]:
            for i in range(len(tool_calls) - seq_len + 1):
                tool_seq = tuple(tool_calls[j][1] for j in range(i, i + seq_len))
                
                # Get context for this sequence
                context = tool_calls[i][2]
                
                # Detect outcome
                outcome = 'unknown'
                for j in range(i, min(i + seq_len + 3, len(tool_calls))):
                    if self.patterns['success'].search(tool_calls[j][2]):
                        outcome = 'success'
                        break
                    elif self.patterns['failure'].search(tool_calls[j][2]):
                        outcome = 'failure'
                        break
                
                sequences[tool_seq]['count'] += 1
                sequences[tool_seq]['contexts'].append(context[:100])
                sequences[tool_seq]['outcomes'].append(outcome)
        
        # Convert to ToolSequence objects
        tool_sequences = []
        for seq, data in sorted(sequences.items(), key=lambda x: x[1]['count'], reverse=True):
            if data['count'] >= 2:  # Only keep sequences that appear at least twice
                most_common_outcome = Counter(data['outcomes']).most_common(1)[0][0]
                context_summary = data['contexts'][0] if data['contexts'] else 'Unknown'
                
                tool_sequences.append(ToolSequence(
                    tools=list(seq),
                    frequency=data['count'],
                    context=context_summary,
                    typical_outcome=most_common_outcome
                ))
        
        print(f"🔗 Found {len(tool_sequences)} common tool sequences")
        return tool_sequences
    
    def generate_cycle_report(self, output_file: str = "DEVELOPMENT_CYCLE.md"):
        """Generate comprehensive development cycle documentation"""
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 🔄 Development Cycle Analysis\n\n")
            f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Source**: {self.input_file}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## 📊 Executive Summary\n\n")
            f.write(f"- **Total Workflow Phases**: {len(self.phases)}\n")
            f.write(f"- **Total Iteration Cycles**: {len(self.cycles)}\n")
            f.write(f"- **Common Tool Sequences**: {len(self.tool_sequences)}\n\n")
            
            # Phase Distribution
            phase_counts = Counter(p.phase_type for p in self.phases)
            f.write("### Phase Distribution\n\n")
            for phase, count in phase_counts.most_common():
                percentage = (count / len(self.phases)) * 100 if self.phases else 0
                f.write(f"- **{phase}**: {count} occurrences ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Success Rate
            successful_cycles = sum(1 for c in self.cycles if c.final_outcome == 'success')
            success_rate = (successful_cycles / len(self.cycles)) * 100 if self.cycles else 0
            f.write(f"### Success Rate: {success_rate:.1f}%\n\n")
            f.write(f"- **Successful Cycles**: {successful_cycles}\n")
            f.write(f"- **Failed Cycles**: {len(self.cycles) - successful_cycles}\n\n")
            
            # Average Iterations
            avg_iterations = sum(c.iterations_count for c in self.cycles) / len(self.cycles) if self.cycles else 0
            f.write(f"### Average Iterations per Cycle: {avg_iterations:.1f}\n\n")
            
            f.write("---\n\n")
            
            # Workflow Phases Detail
            f.write("## 🔄 Workflow Phases\n\n")
            for i, phase in enumerate(self.phases[:20], 1):  # Limit to first 20
                f.write(f"### Phase #{i}: {phase.phase_type.replace('_', ' ').title()}\n\n")
                f.write(f"- **Lines**: {phase.start_line} - {phase.end_line} ({phase.duration_lines} lines)\n")
                f.write(f"- **Outcome**: {phase.outcome}\n")
                if phase.tools_used:
                    f.write(f"- **Tools Used**: {', '.join(phase.tools_used)}\n")
                if phase.actions:
                    f.write(f"- **Key Actions**:\n")
                    for action in phase.actions[:3]:
                        f.write(f"  - {action[:80]}...\n")
                f.write("\n")
            
            f.write("---\n\n")
            
            # Iteration Cycles Detail
            f.write("## 🔁 Iteration Cycles\n\n")
            for cycle in self.cycles[:10]:  # Limit to first 10
                f.write(f"### Cycle #{cycle.cycle_id}\n\n")
                f.write(f"- **Lines**: {cycle.start_line} - {cycle.end_line}\n")
                f.write(f"- **Iterations**: {cycle.iterations_count}\n")
                f.write(f"- **Final Outcome**: {cycle.final_outcome}\n")
                f.write(f"- **Phases**: {len(cycle.phases)}\n")
                f.write(f"- **Tool Sequence**: {' → '.join(cycle.tools_sequence[:10])}\n")
                
                if cycle.key_decisions:
                    f.write(f"\n**Key Decisions**:\n")
                    for decision in cycle.key_decisions[:3]:
                        f.write(f"- {decision}\n")
                
                f.write("\n---\n\n")
            
            # Common Tool Sequences
            f.write("## 🔗 Common Tool Sequences\n\n")
            f.write("These are the most frequent tool call patterns:\n\n")
            
            for i, seq in enumerate(self.tool_sequences[:15], 1):
                f.write(f"### Pattern #{i} (Used {seq.frequency} times)\n\n")
                f.write(f"**Sequence**: {' → '.join(seq.tools)}\n\n")
                f.write(f"**Typical Outcome**: {seq.typical_outcome}\n\n")
                f.write(f"**Context**: {seq.context}...\n\n")
                f.write("---\n\n")
            
            # Formalized Development Cycle
            f.write("## 📋 Formalized Development Cycle\n\n")
            f.write("Based on the analysis, here is the recommended development cycle:\n\n")
            
            f.write("```\n")
            f.write("DEVELOPMENT_CYCLE = {\n")
            f.write("    'phases': [\n")
            for phase, count in phase_counts.most_common():
                f.write(f"        '{phase}',  # Occurs {count} times\n")
            f.write("    ],\n\n")
            
            f.write("    'typical_tool_sequence': [\n")
            if self.tool_sequences:
                most_common = self.tool_sequences[0].tools
                for tool in most_common:
                    f.write(f"        '{tool}',\n")
            f.write("    ],\n\n")
            
            f.write(f"    'average_iterations': {avg_iterations:.1f},\n")
            f.write(f"    'success_rate': {success_rate:.1f},\n\n")
            
            f.write("    'best_practices': [\n")
            f.write("        'Always gather context before implementing',\n")
            f.write("        'Test after each significant change',\n")
            f.write("        'Commit working code before major refactors',\n")
            f.write("        'Use quick tests before full validation',\n")
            f.write("        'Document decisions and reasoning',\n")
            f.write("    ]\n")
            f.write("}\n")
            f.write("```\n\n")
            
            # Workflow Diagram
            f.write("## 🌊 Workflow Diagram\n\n")
            f.write("```mermaid\n")
            f.write("flowchart TD\n")
            f.write("    Start([User Request/Error]) --> Context[Context Gathering]\n")
            f.write("    Context --> Research{Need Research?}\n")
            f.write("    Research -->|Yes| Research_Phase[Research Phase]\n")
            f.write("    Research -->|No| Analysis[Analysis Phase]\n")
            f.write("    Research_Phase --> Analysis\n")
            f.write("    Analysis --> Planning[Planning & Design]\n")
            f.write("    Planning --> Implementation[Implementation]\n")
            f.write("    Implementation --> Testing[Testing]\n")
            f.write("    Testing --> Success{Success?}\n")
            f.write("    Success -->|Yes| Validation[Final Validation]\n")
            f.write("    Success -->|No| Debug[Debugging]\n")
            f.write("    Debug --> Analysis\n")
            f.write("    Validation --> Complete([Complete])\n")
            f.write("```\n\n")
            
            f.write("---\n\n")
            f.write("## 🎯 Recommendations\n\n")
            f.write("1. **Context First**: Always gather context before implementation\n")
            f.write(f"2. **Iterate Smart**: Average {avg_iterations:.0f} iterations is normal, don't give up\n")
            f.write("3. **Test Early**: Run tests after each significant change\n")
            f.write("4. **Use Patterns**: Leverage common tool sequences for efficiency\n")
            f.write(f"5. **Learn from Failures**: {len(self.cycles) - successful_cycles} failed cycles provide learning opportunities\n")
        
        print(f"✅ Development cycle report saved to {output_path}")
    
    def generate_json_export(self, output_file: str = "development_cycle.json"):
        """Export analysis data as JSON"""
        data = {
            'metadata': {
                'source_file': str(self.input_file),
                'analysis_date': datetime.now().isoformat(),
                'total_lines': len(self.lines)
            },
            'phases': [asdict(p) for p in self.phases],
            'cycles': [
                {
                    **asdict(c),
                    'phases': [asdict(p) for p in c.phases]
                }
                for c in self.cycles
            ],
            'tool_sequences': [asdict(s) for s in self.tool_sequences],
            'statistics': {
                'total_phases': len(self.phases),
                'total_cycles': len(self.cycles),
                'success_rate': (sum(1 for c in self.cycles if c.final_outcome == 'success') / len(self.cycles) * 100) if self.cycles else 0,
                'avg_iterations': sum(c.iterations_count for c in self.cycles) / len(self.cycles) if self.cycles else 0
            }
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON export saved to {output_path}")
    
    def run(self):
        """Main analysis workflow"""
        print("\n🚀 Starting Development Cycle Analysis\n")
        print("=" * 60)
        
        # Load content
        self.load_content()
        
        # Detect tool calls
        print("\n🔧 Detecting tool calls...")
        tool_calls = self.detect_tool_calls()
        
        # Detect workflow phases
        print("\n🔄 Detecting workflow phases...")
        self.phases = self.detect_workflow_phases()
        
        # Detect iteration cycles
        print("\n🔁 Detecting iteration cycles...")
        self.cycles = self.detect_iteration_cycles(tool_calls)
        
        # Analyze tool sequences
        print("\n🔗 Analyzing tool sequences...")
        self.tool_sequences = self.analyze_tool_sequences(tool_calls)
        
        # Generate outputs
        print("\n💾 Generating outputs...")
        self.generate_cycle_report()
        self.generate_json_export()
        
        # Print summary
        print("\n" + "=" * 60)
        print("\n✨ ANALYSIS COMPLETE!\n")
        print(f"📊 Workflow Phases: {len(self.phases)}")
        print(f"🔁 Iteration Cycles: {len(self.cycles)}")
        print(f"🔗 Tool Sequences: {len(self.tool_sequences)}")
        
        if self.cycles:
            successful = sum(1 for c in self.cycles if c.final_outcome == 'success')
            success_rate = (successful / len(self.cycles)) * 100
            print(f"\n✅ Success Rate: {success_rate:.1f}%")
            
            avg_iterations = sum(c.iterations_count for c in self.cycles) / len(self.cycles)
            print(f"🔄 Avg Iterations: {avg_iterations:.1f}")
        
        print("\n📁 Output files created:")
        print("   - DEVELOPMENT_CYCLE.md (formalized cycle documentation)")
        print("   - development_cycle.json (structured analysis data)")
        print("\n🎯 Done!\n")


if __name__ == "__main__":
    extractor = DevelopmentCycleExtractor("copilot.md")
    extractor.run()
