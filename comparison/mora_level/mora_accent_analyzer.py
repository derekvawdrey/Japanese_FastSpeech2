#!/usr/bin/env python3
"""
Mora-Level Pitch Accent Analyzer for Japanese

This module provides linguistically-motivated pitch accent analysis by:
1. Segmenting audio into morae (not just phonemes)
2. Extracting F0 (pitch) per mora
3. Classifying each mora as HIGH (H) or LOW (L)
4. Detecting accent patterns (heiban, atamadaka, nakadaka, odaka)
5. Comparing patterns against reference or dictionary

Japanese Pitch Accent Types:
- Type 0 (平板型/heiban): No drop, pattern is LH...H (flat/unaccented)
- Type 1 (頭高型/atamadaka): Drop after mora 1, pattern is HL...L
- Type n (中高型/nakadaka): Drop after mora n (2 <= n < last)
- Type n=last (尾高型/odaka): Drop after last mora (visible with particles)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import parselmouth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Mappings
# ============================================================================

# Julius phonemes that form the nucleus of a mora (vowels)
VOWELS = {'a', 'i', 'u', 'e', 'o'}

# Special morae
MORAIC_NASAL = 'N'  # ん
GEMINATE = 'q'      # っ (represented as 'cl' in some systems)
LONG_VOWEL_MARKERS = {':', 'H'}  # Long vowel marker

# Julius phoneme to mora mapping
# Consonants that combine with following vowel to form one mora
CONSONANTS = {
    'k', 'g', 's', 'z', 'sh', 'j', 't', 'd', 'ch', 'ts', 
    'n', 'h', 'f', 'b', 'p', 'm', 'y', 'r', 'w',
    'ky', 'gy', 'sy', 'zy', 'ty', 'dy', 'ny', 'hy', 'by', 'py', 'my', 'ry',
    'kw', 'gw'
}

# Phonemes to skip (silence, etc.)
SKIP_PHONEMES = {'pau', 'sil', 'sp', '', 'cl'}


class AccentType(Enum):
    """Japanese pitch accent types"""
    HEIBAN = 0       # 平板型 - No accent (LH...H)
    ATAMADAKA = 1    # 頭高型 - Accent on first mora (HL...L)
    NAKADAKA = 2     # 中高型 - Accent in middle (LH...HL...L)
    ODAKA = -1       # 尾高型 - Accent on last mora (LH...HL) - only visible with particle


class PitchLevel(Enum):
    """Pitch level for a mora"""
    HIGH = 'H'
    LOW = 'L'
    UNKNOWN = '?'


@dataclass
class Mora:
    """Represents a single Japanese mora"""
    text: str                          # The mora text (e.g., "か", "きょ")
    phonemes: List[str]                # Julius phonemes making up this mora
    start_time: float = 0.0            # Start time in seconds
    end_time: float = 0.0              # End time in seconds
    f0_values: List[float] = field(default_factory=list)  # F0 samples within this mora
    mean_f0: Optional[float] = None    # Mean F0 for this mora
    pitch_level: PitchLevel = PitchLevel.UNKNOWN  # H or L classification
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def __repr__(self):
        return f"Mora({self.text}, {self.pitch_level.value}, f0={self.mean_f0:.1f}Hz)" if self.mean_f0 else f"Mora({self.text})"


@dataclass
class AccentPattern:
    """Represents a detected accent pattern"""
    pattern_string: str          # e.g., "LHL" or "LHH"
    accent_type: AccentType      # The accent type
    accent_position: int         # Position of accent (0 for heiban)
    morae: List[Mora]            # The analyzed morae
    confidence: float = 0.0      # Confidence score (0-1)
    
    def __repr__(self):
        type_name = self.accent_type.name
        return f"AccentPattern({self.pattern_string}, type={type_name}, pos={self.accent_position})"


@dataclass 
class AccentComparisonResult:
    """Result of comparing two accent patterns"""
    reference_pattern: AccentPattern
    test_pattern: AccentPattern
    patterns_match: bool              # Do the H/L patterns match?
    accent_type_match: bool           # Do the accent types match?
    accent_position_match: bool       # Do the accent positions match?
    mora_accuracy: float              # Percentage of morae with matching H/L
    detailed_comparison: List[Tuple[str, str, bool]]  # Per-mora comparison
    
    @property
    def is_correct(self) -> bool:
        """Returns True if the accent is considered correct"""
        return self.accent_type_match and self.accent_position_match


# ============================================================================
# Mora Segmentation
# ============================================================================

def julius_phonemes_to_morae(phonemes: List[Tuple[float, float, str]]) -> List[Mora]:
    """
    Convert Julius phoneme segments to mora segments.
    
    Japanese mora structure:
    - (C)(y)V: Optional consonant, optional glide, required vowel
    - N: Moraic nasal (ん)
    - Q: Geminate consonant (っ)
    - Long vowels count as separate morae
    
    Args:
        phonemes: List of (start_time, end_time, phoneme) tuples
        
    Returns:
        List of Mora objects
    """
    morae = []
    i = 0
    
    while i < len(phonemes):
        start_time, end_time, phoneme = phonemes[i]
        
        # Skip silence/pause markers
        if phoneme.lower() in SKIP_PHONEMES:
            i += 1
            continue
        
        # Check for moraic nasal (ん)
        if phoneme.upper() == 'N' or phoneme == 'N':
            mora = Mora(
                text='ん',
                phonemes=[phoneme],
                start_time=start_time,
                end_time=end_time
            )
            morae.append(mora)
            i += 1
            continue
        
        # Check for geminate (っ) - often marked as 'cl' or 'q'
        if phoneme.lower() in {'cl', 'q'}:
            mora = Mora(
                text='っ',
                phonemes=[phoneme],
                start_time=start_time,
                end_time=end_time
            )
            morae.append(mora)
            i += 1
            continue
        
        # Check if this is a vowel (standalone mora)
        if phoneme.lower() in VOWELS:
            mora = Mora(
                text=_phoneme_to_kana(phoneme),
                phonemes=[phoneme],
                start_time=start_time,
                end_time=end_time
            )
            morae.append(mora)
            i += 1
            continue
        
        # Check for long vowel marker
        if phoneme in LONG_VOWEL_MARKERS or phoneme.endswith(':'):
            # Long vowel - counts as separate mora
            mora = Mora(
                text='ー',
                phonemes=[phoneme],
                start_time=start_time,
                end_time=end_time
            )
            morae.append(mora)
            i += 1
            continue
        
        # This should be a consonant - look for following vowel
        consonant = phoneme
        consonant_start = start_time
        
        # Check if next phoneme is a vowel
        if i + 1 < len(phonemes):
            next_start, next_end, next_phoneme = phonemes[i + 1]
            
            if next_phoneme.lower() in VOWELS:
                # C + V = one mora
                mora = Mora(
                    text=_phonemes_to_kana(consonant, next_phoneme),
                    phonemes=[consonant, next_phoneme],
                    start_time=consonant_start,
                    end_time=next_end
                )
                morae.append(mora)
                i += 2
                continue
            elif next_phoneme.lower() in {'y'} and i + 2 < len(phonemes):
                # Check for CyV pattern (きょ, しゃ, etc.)
                third_start, third_end, third_phoneme = phonemes[i + 2]
                if third_phoneme.lower() in VOWELS:
                    mora = Mora(
                        text=_phonemes_to_kana(consonant + 'y', third_phoneme),
                        phonemes=[consonant, next_phoneme, third_phoneme],
                        start_time=consonant_start,
                        end_time=third_end
                    )
                    morae.append(mora)
                    i += 3
                    continue
        
        # Fallback: treat as single unit
        mora = Mora(
            text=f"[{phoneme}]",
            phonemes=[phoneme],
            start_time=start_time,
            end_time=end_time
        )
        morae.append(mora)
        i += 1
    
    return morae


def _phoneme_to_kana(vowel: str) -> str:
    """Convert a single vowel phoneme to hiragana"""
    mapping = {'a': 'あ', 'i': 'い', 'u': 'う', 'e': 'え', 'o': 'お'}
    return mapping.get(vowel.lower(), vowel)


def _phonemes_to_kana(consonant: str, vowel: str) -> str:
    """Convert consonant + vowel to hiragana"""
    # Simplified mapping - covers common cases
    kana_table = {
        ('k', 'a'): 'か', ('k', 'i'): 'き', ('k', 'u'): 'く', ('k', 'e'): 'け', ('k', 'o'): 'こ',
        ('g', 'a'): 'が', ('g', 'i'): 'ぎ', ('g', 'u'): 'ぐ', ('g', 'e'): 'げ', ('g', 'o'): 'ご',
        ('s', 'a'): 'さ', ('s', 'i'): 'し', ('s', 'u'): 'す', ('s', 'e'): 'せ', ('s', 'o'): 'そ',
        ('sh', 'i'): 'し', ('sh', 'a'): 'しゃ', ('sh', 'u'): 'しゅ', ('sh', 'o'): 'しょ',
        ('z', 'a'): 'ざ', ('z', 'i'): 'じ', ('z', 'u'): 'ず', ('z', 'e'): 'ぜ', ('z', 'o'): 'ぞ',
        ('j', 'a'): 'じゃ', ('j', 'i'): 'じ', ('j', 'u'): 'じゅ', ('j', 'o'): 'じょ',
        ('t', 'a'): 'た', ('t', 'i'): 'ち', ('t', 'u'): 'つ', ('t', 'e'): 'て', ('t', 'o'): 'と',
        ('ch', 'i'): 'ち', ('ch', 'a'): 'ちゃ', ('ch', 'u'): 'ちゅ', ('ch', 'o'): 'ちょ',
        ('ts', 'u'): 'つ',
        ('d', 'a'): 'だ', ('d', 'i'): 'ぢ', ('d', 'u'): 'づ', ('d', 'e'): 'で', ('d', 'o'): 'ど',
        ('n', 'a'): 'な', ('n', 'i'): 'に', ('n', 'u'): 'ぬ', ('n', 'e'): 'ね', ('n', 'o'): 'の',
        ('h', 'a'): 'は', ('h', 'i'): 'ひ', ('h', 'u'): 'ふ', ('h', 'e'): 'へ', ('h', 'o'): 'ほ',
        ('f', 'u'): 'ふ',
        ('b', 'a'): 'ば', ('b', 'i'): 'び', ('b', 'u'): 'ぶ', ('b', 'e'): 'べ', ('b', 'o'): 'ぼ',
        ('p', 'a'): 'ぱ', ('p', 'i'): 'ぴ', ('p', 'u'): 'ぷ', ('p', 'e'): 'ぺ', ('p', 'o'): 'ぽ',
        ('m', 'a'): 'ま', ('m', 'i'): 'み', ('m', 'u'): 'む', ('m', 'e'): 'め', ('m', 'o'): 'も',
        ('y', 'a'): 'や', ('y', 'u'): 'ゆ', ('y', 'o'): 'よ',
        ('r', 'a'): 'ら', ('r', 'i'): 'り', ('r', 'u'): 'る', ('r', 'e'): 'れ', ('r', 'o'): 'ろ',
        ('w', 'a'): 'わ', ('w', 'o'): 'を',
        # Palatalized consonants
        ('ky', 'a'): 'きゃ', ('ky', 'u'): 'きゅ', ('ky', 'o'): 'きょ',
        ('gy', 'a'): 'ぎゃ', ('gy', 'u'): 'ぎゅ', ('gy', 'o'): 'ぎょ',
        ('ny', 'a'): 'にゃ', ('ny', 'u'): 'にゅ', ('ny', 'o'): 'にょ',
        ('hy', 'a'): 'ひゃ', ('hy', 'u'): 'ひゅ', ('hy', 'o'): 'ひょ',
        ('by', 'a'): 'びゃ', ('by', 'u'): 'びゅ', ('by', 'o'): 'びょ',
        ('py', 'a'): 'ぴゃ', ('py', 'u'): 'ぴゅ', ('py', 'o'): 'ぴょ',
        ('my', 'a'): 'みゃ', ('my', 'u'): 'みゅ', ('my', 'o'): 'みょ',
        ('ry', 'a'): 'りゃ', ('ry', 'u'): 'りゅ', ('ry', 'o'): 'りょ',
    }
    
    key = (consonant.lower(), vowel.lower())
    return kana_table.get(key, f"{consonant}{vowel}")


# ============================================================================
# F0 Extraction and Analysis
# ============================================================================

class MoraAccentAnalyzer:
    """
    Analyzes pitch accent at the mora level from audio files.
    """
    
    def __init__(
        self,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 500.0,
        time_step: float = 0.005,
        high_low_threshold: float = 0.5,  # Semitones difference for H/L boundary
    ):
        """
        Initialize the analyzer.
        
        Args:
            pitch_floor: Minimum pitch to detect (Hz)
            pitch_ceiling: Maximum pitch to detect (Hz)
            time_step: Time step for pitch analysis (seconds)
            high_low_threshold: Threshold in semitones for H/L classification
        """
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.time_step = time_step
        self.high_low_threshold = high_low_threshold
    
    def analyze_audio(
        self,
        wav_path: str,
        phonemes: List[Tuple[float, float, str]],
    ) -> AccentPattern:
        """
        Analyze pitch accent pattern from audio file.
        
        Args:
            wav_path: Path to WAV file
            phonemes: List of (start, end, phoneme) tuples from forced alignment
            
        Returns:
            AccentPattern with detected pattern
        """
        # Load audio and extract pitch
        sound = parselmouth.Sound(wav_path)
        pitch = sound.to_pitch(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling
        )
        
        # Convert phonemes to morae
        morae = julius_phonemes_to_morae(phonemes)
        
        if not morae:
            logger.warning("No morae detected from phonemes")
            return AccentPattern(
                pattern_string="",
                accent_type=AccentType.HEIBAN,
                accent_position=0,
                morae=[],
                confidence=0.0
            )
        
        # Extract F0 for each mora
        self._extract_f0_per_mora(pitch, morae)
        
        # Classify each mora as H or L
        self._classify_pitch_levels(morae)
        
        # Detect accent pattern
        pattern = self._detect_accent_pattern(morae)
        
        return pattern
    
    def _extract_f0_per_mora(
        self,
        pitch: parselmouth.Pitch,
        morae: List[Mora]
    ) -> None:
        """Extract F0 values for each mora from pitch object"""
        pitch_times = pitch.xs()
        pitch_values = pitch.selected_array['frequency']
        
        for mora in morae:
            # Find pitch frames within this mora's time range
            mask = (pitch_times >= mora.start_time) & (pitch_times <= mora.end_time)
            mora_f0 = pitch_values[mask]
            
            # Filter out unvoiced frames (F0 = 0)
            voiced_f0 = mora_f0[mora_f0 > 0]
            
            if len(voiced_f0) > 0:
                mora.f0_values = voiced_f0.tolist()
                mora.mean_f0 = float(np.mean(voiced_f0))
            else:
                mora.f0_values = []
                mora.mean_f0 = None
    
    def _classify_pitch_levels(self, morae: List[Mora]) -> None:
        """
        Classify each mora as HIGH or LOW based on relative pitch.
        
        Japanese pitch accent rule: The first mora has opposite pitch from second mora.
        - If word is accented (type > 0): First mora is opposite of second
        - Pattern typically: LH...HL... or HL...
        
        We use relative pitch changes between adjacent morae to classify.
        """
        if not morae:
            return
        
        # Get morae with valid F0
        valid_morae = [m for m in morae if m.mean_f0 is not None]
        
        if len(valid_morae) < 2:
            # Can't determine pattern with less than 2 morae
            for m in valid_morae:
                m.pitch_level = PitchLevel.HIGH  # Default
            return
        
        # Calculate mean F0 across all morae for reference
        all_f0 = [m.mean_f0 for m in valid_morae]
        mean_f0 = np.mean(all_f0)
        
        # Convert to semitones relative to mean
        semitones = [12 * np.log2(f0 / mean_f0) if f0 > 0 else 0 for f0 in all_f0]
        
        # Use relative pitch changes to classify
        # In Japanese, the key is identifying the pitch DROP
        for i, mora in enumerate(valid_morae):
            idx = morae.index(mora)
            
            if semitones[i] > self.high_low_threshold:
                morae[idx].pitch_level = PitchLevel.HIGH
            elif semitones[i] < -self.high_low_threshold:
                morae[idx].pitch_level = PitchLevel.LOW
            else:
                # Near mean - need to look at context
                if i == 0:
                    # First mora - compare with second
                    if len(semitones) > 1:
                        morae[idx].pitch_level = PitchLevel.LOW if semitones[1] > semitones[0] else PitchLevel.HIGH
                    else:
                        morae[idx].pitch_level = PitchLevel.HIGH
                else:
                    # Compare with previous mora
                    if semitones[i] < semitones[i-1] - 0.5:
                        morae[idx].pitch_level = PitchLevel.LOW
                    elif semitones[i] > semitones[i-1] + 0.5:
                        morae[idx].pitch_level = PitchLevel.HIGH
                    else:
                        # Carry over from previous
                        morae[idx].pitch_level = valid_morae[i-1].pitch_level
        
        # Handle morae with no F0 (unvoiced)
        for i, mora in enumerate(morae):
            if mora.mean_f0 is None:
                # Interpolate from neighbors
                prev_level = morae[i-1].pitch_level if i > 0 else PitchLevel.LOW
                next_level = morae[i+1].pitch_level if i < len(morae)-1 else prev_level
                mora.pitch_level = prev_level  # Default to previous
    
    def _detect_accent_pattern(self, morae: List[Mora]) -> AccentPattern:
        """
        Detect the accent pattern type from classified morae.
        
        Returns AccentPattern with type and position.
        """
        if not morae:
            return AccentPattern(
                pattern_string="",
                accent_type=AccentType.HEIBAN,
                accent_position=0,
                morae=[],
                confidence=0.0
            )
        
        # Build pattern string
        pattern_string = ''.join(m.pitch_level.value for m in morae)
        
        # Find accent position (where H drops to L)
        accent_position = 0
        for i in range(len(morae) - 1):
            if morae[i].pitch_level == PitchLevel.HIGH and morae[i+1].pitch_level == PitchLevel.LOW:
                accent_position = i + 1  # 1-indexed position
                break
        
        # Determine accent type
        if accent_position == 0:
            # No drop found - could be heiban (LH...H) or undetected
            if len(morae) >= 2 and morae[0].pitch_level == PitchLevel.LOW:
                accent_type = AccentType.HEIBAN
            else:
                accent_type = AccentType.HEIBAN  # Default
        elif accent_position == 1:
            accent_type = AccentType.ATAMADAKA
        elif accent_position == len(morae):
            accent_type = AccentType.ODAKA
        else:
            accent_type = AccentType.NAKADAKA
        
        # Calculate confidence based on F0 variance and pattern clarity
        confidence = self._calculate_confidence(morae)
        
        return AccentPattern(
            pattern_string=pattern_string,
            accent_type=accent_type,
            accent_position=accent_position,
            morae=morae,
            confidence=confidence
        )
    
    def _calculate_confidence(self, morae: List[Mora]) -> float:
        """Calculate confidence score for the detected pattern"""
        if not morae:
            return 0.0
        
        # Factors that increase confidence:
        # 1. Clear F0 difference between H and L morae
        # 2. Consistent pattern
        # 3. Voiced morae
        
        voiced_count = sum(1 for m in morae if m.mean_f0 is not None)
        voiced_ratio = voiced_count / len(morae)
        
        # Get H and L F0 values
        h_f0 = [m.mean_f0 for m in morae if m.pitch_level == PitchLevel.HIGH and m.mean_f0]
        l_f0 = [m.mean_f0 for m in morae if m.pitch_level == PitchLevel.LOW and m.mean_f0]
        
        if h_f0 and l_f0:
            # Calculate separation in semitones
            h_mean = np.mean(h_f0)
            l_mean = np.mean(l_f0)
            separation = abs(12 * np.log2(h_mean / l_mean))
            separation_score = min(separation / 3.0, 1.0)  # 3 semitones = full score
        else:
            separation_score = 0.5
        
        confidence = (voiced_ratio * 0.4 + separation_score * 0.6)
        return round(confidence, 3)
    
    def compare_patterns(
        self,
        reference: AccentPattern,
        test: AccentPattern
    ) -> AccentComparisonResult:
        """
        Compare two accent patterns.
        
        Args:
            reference: The reference/correct pattern
            test: The pattern to test
            
        Returns:
            AccentComparisonResult with detailed comparison
        """
        # Compare pattern strings
        ref_pattern = reference.pattern_string
        test_pattern = test.pattern_string
        
        # Handle length differences by padding shorter pattern
        max_len = max(len(ref_pattern), len(test_pattern))
        ref_padded = ref_pattern.ljust(max_len, '?')
        test_padded = test_pattern.ljust(max_len, '?')
        
        # Per-mora comparison
        detailed = []
        matches = 0
        for r, t in zip(ref_padded, test_padded):
            match = (r == t) or r == '?' or t == '?'
            detailed.append((r, t, match))
            if match and r != '?':
                matches += 1
        
        valid_comparisons = sum(1 for r, t, _ in detailed if r != '?' and t != '?')
        mora_accuracy = matches / valid_comparisons if valid_comparisons > 0 else 0.0
        
        return AccentComparisonResult(
            reference_pattern=reference,
            test_pattern=test,
            patterns_match=(ref_pattern == test_pattern),
            accent_type_match=(reference.accent_type == test.accent_type),
            accent_position_match=(reference.accent_position == test.accent_position),
            mora_accuracy=mora_accuracy,
            detailed_comparison=detailed
        )


# ============================================================================
# Dictionary-based accent lookup
# ============================================================================

# Common accent patterns (simplified - in production, use full dictionary)
ACCENT_DICTIONARY: Dict[str, int] = {
    # Format: word -> accent position (0 = heiban)
    'なんで': 1,  # HL - atamadaka
    'ありがとう': 0,  # LHHHH - heiban (in standard Japanese)
    'こんにちは': 0,  # LHHHH - heiban
    'さようなら': 0,  # LHHHH - heiban  
    'おはよう': 0,  # LHHH - heiban
    'すみません': 0,  # LHHHH - heiban
    'わたし': 0,  # LHH - heiban
    'あなた': 2,  # LHL - nakadaka
    'これ': 0,  # LH - heiban
    'それ': 0,  # LH - heiban
    'あれ': 0,  # LH - heiban
    'ここ': 0,  # LH - heiban
    'そこ': 0,  # LH - heiban
    'あそこ': 0,  # LHH - heiban
    'いま': 1,  # HL - atamadaka
    'きょう': 1,  # HL - atamadaka
    'あした': 0,  # LHH - heiban (or 3 in some dialects)
    'きのう': 2,  # LHL - nakadaka
}


def get_expected_accent(word: str) -> Optional[int]:
    """
    Look up expected accent position for a word.
    
    Returns accent position (0 = heiban, 1+ = accent position) or None if unknown.
    """
    return ACCENT_DICTIONARY.get(word)


def pattern_from_accent_position(num_morae: int, accent_pos: int) -> str:
    """
    Generate expected H/L pattern from accent position.
    
    Args:
        num_morae: Number of morae in the word
        accent_pos: Accent position (0 = heiban)
        
    Returns:
        Pattern string like "LHH" or "LHL"
    """
    if num_morae <= 0:
        return ""
    
    if num_morae == 1:
        return "H" if accent_pos == 1 else "L"
    
    if accent_pos == 0:
        # Heiban: LH...H
        return "L" + "H" * (num_morae - 1)
    elif accent_pos == 1:
        # Atamadaka: HL...L
        return "H" + "L" * (num_morae - 1)
    else:
        # Nakadaka/Odaka: LH...HL...L
        pattern = "L" + "H" * (accent_pos - 1) + "L" * (num_morae - accent_pos)
        return pattern


# ============================================================================
# High-level comparison functions
# ============================================================================

def compare_audio_accent(
    reference_wav: str,
    reference_phonemes: List[Tuple[float, float, str]],
    test_wav: str,
    test_phonemes: List[Tuple[float, float, str]],
    analyzer: Optional[MoraAccentAnalyzer] = None
) -> AccentComparisonResult:
    """
    Compare pitch accent between reference and test audio.
    
    Args:
        reference_wav: Path to reference audio
        reference_phonemes: Phoneme segments for reference
        test_wav: Path to test audio
        test_phonemes: Phoneme segments for test
        analyzer: Optional analyzer instance
        
    Returns:
        AccentComparisonResult
    """
    if analyzer is None:
        analyzer = MoraAccentAnalyzer()
    
    ref_pattern = analyzer.analyze_audio(reference_wav, reference_phonemes)
    test_pattern = analyzer.analyze_audio(test_wav, test_phonemes)
    
    return analyzer.compare_patterns(ref_pattern, test_pattern)


def compare_audio_to_dictionary(
    test_wav: str,
    test_phonemes: List[Tuple[float, float, str]],
    word: str,
    expected_accent: Optional[int] = None,
    analyzer: Optional[MoraAccentAnalyzer] = None
) -> AccentComparisonResult:
    """
    Compare audio pitch accent against dictionary entry.
    
    Args:
        test_wav: Path to test audio
        test_phonemes: Phoneme segments
        word: The word being pronounced
        expected_accent: Expected accent position (or None to lookup)
        analyzer: Optional analyzer instance
        
    Returns:
        AccentComparisonResult
    """
    if analyzer is None:
        analyzer = MoraAccentAnalyzer()
    
    # Get test pattern from audio
    test_pattern = analyzer.analyze_audio(test_wav, test_phonemes)
    
    # Get expected pattern
    if expected_accent is None:
        expected_accent = get_expected_accent(word)
    
    if expected_accent is None:
        logger.warning(f"No dictionary entry for '{word}', using test pattern as reference")
        return AccentComparisonResult(
            reference_pattern=test_pattern,
            test_pattern=test_pattern,
            patterns_match=True,
            accent_type_match=True,
            accent_position_match=True,
            mora_accuracy=1.0,
            detailed_comparison=[(p, p, True) for p in test_pattern.pattern_string]
        )
    
    # Build expected pattern
    num_morae = len(test_pattern.morae)
    expected_pattern_str = pattern_from_accent_position(num_morae, expected_accent)
    
    # Determine expected accent type
    if expected_accent == 0:
        expected_type = AccentType.HEIBAN
    elif expected_accent == 1:
        expected_type = AccentType.ATAMADAKA
    elif expected_accent == num_morae:
        expected_type = AccentType.ODAKA
    else:
        expected_type = AccentType.NAKADAKA
    
    # Create reference pattern object
    reference_pattern = AccentPattern(
        pattern_string=expected_pattern_str,
        accent_type=expected_type,
        accent_position=expected_accent,
        morae=[],  # No actual morae for dictionary lookup
        confidence=1.0
    )
    
    return analyzer.compare_patterns(reference_pattern, test_pattern)


# ============================================================================
# Visualization helpers
# ============================================================================

def visualize_accent_pattern(pattern: AccentPattern) -> str:
    """
    Create ASCII visualization of accent pattern.
    
    Returns something like:
        か き く
         H  H  L
         ──┐
           └──
    """
    if not pattern.morae:
        return f"Pattern: {pattern.pattern_string}"
    
    lines = []
    
    # Mora text line
    mora_line = "  ".join(f"{m.text:^3}" for m in pattern.morae)
    lines.append(mora_line)
    
    # H/L line
    hl_line = "  ".join(f" {m.pitch_level.value} " for m in pattern.morae)
    lines.append(hl_line)
    
    # Pitch contour line (ASCII art)
    contour = []
    for i, mora in enumerate(pattern.morae):
        if mora.pitch_level == PitchLevel.HIGH:
            contour.append("──")
        else:
            contour.append("__")
        
        if i < len(pattern.morae) - 1:
            curr = pattern.morae[i].pitch_level
            next_p = pattern.morae[i + 1].pitch_level
            if curr == PitchLevel.HIGH and next_p == PitchLevel.LOW:
                contour.append("╲")
            elif curr == PitchLevel.LOW and next_p == PitchLevel.HIGH:
                contour.append("╱")
            else:
                contour.append("─")
    
    lines.append("".join(contour))
    
    # Info line
    info = f"Type: {pattern.accent_type.name}, Position: {pattern.accent_position}, Confidence: {pattern.confidence:.0%}"
    lines.append(info)
    
    return "\n".join(lines)


def format_comparison_result(result: AccentComparisonResult) -> str:
    """Format comparison result for display"""
    lines = []
    lines.append("=" * 60)
    lines.append("ACCENT COMPARISON RESULT")
    lines.append("=" * 60)
    
    lines.append(f"\nReference Pattern: {result.reference_pattern.pattern_string}")
    lines.append(f"Test Pattern:      {result.test_pattern.pattern_string}")
    
    lines.append(f"\nPatterns Match:    {'✓' if result.patterns_match else '✗'}")
    lines.append(f"Accent Type Match: {'✓' if result.accent_type_match else '✗'}")
    lines.append(f"Position Match:    {'✓' if result.accent_position_match else '✗'}")
    lines.append(f"Mora Accuracy:     {result.mora_accuracy:.1%}")
    
    lines.append(f"\nOverall: {'CORRECT ✓' if result.is_correct else 'INCORRECT ✗'}")
    
    # Detailed per-mora comparison
    lines.append("\nPer-mora comparison:")
    for i, (ref, test, match) in enumerate(result.detailed_comparison):
        status = "✓" if match else "✗"
        lines.append(f"  Mora {i+1}: ref={ref} test={test} {status}")
    
    return "\n".join(lines)


# ============================================================================
# Main / Demo
# ============================================================================

if __name__ == "__main__":
    # Demo usage
    print("Mora-Level Pitch Accent Analyzer")
    print("=" * 60)
    
    # Example: expected patterns for common words
    test_words = ['なんで', 'ありがとう', 'こんにちは', 'あなた', 'いま', 'きのう']
    
    print("\nExpected accent patterns from dictionary:")
    for word in test_words:
        accent_pos = get_expected_accent(word)
        if accent_pos is not None:
            # Estimate mora count from word length (simplified)
            num_morae = len(word)
            pattern = pattern_from_accent_position(num_morae, accent_pos)
            accent_type = "heiban" if accent_pos == 0 else f"type-{accent_pos}"
            print(f"  {word}: {pattern} ({accent_type})")
    
    print("\n" + "=" * 60)
    print("To use with audio files:")
    print("""
    from mora_accent_analyzer import MoraAccentAnalyzer, compare_audio_accent
    
    analyzer = MoraAccentAnalyzer()
    
    # With phoneme segments from forced alignment:
    pattern = analyzer.analyze_audio("audio.wav", phonemes)
    print(visualize_accent_pattern(pattern))
    
    # Compare two audio files:
    result = compare_audio_accent(
        "reference.wav", ref_phonemes,
        "test.wav", test_phonemes
    )
    print(format_comparison_result(result))
    """)
