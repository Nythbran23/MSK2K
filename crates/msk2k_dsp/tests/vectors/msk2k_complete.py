#!/usr/bin/env python3
"""
MSK2K Complete Implementation - Unified Module
MSK modulation with PSK2K protocol for meteor scatter

Based on PSK2K by Klaus von der Heide (DJ5HG)
MSK2K adaptation by Roger Banks (GW4WND)
Date: January 2026

MSK2K uses the same protocol as PSK2K:
- 258-bit packets at 2000 baud (129ms)
- Rate 1/2, K=7 convolutional coding
- Soft-decision Viterbi with LLR accumulation
- Same interleaving and source encoding

But replaces BPSK modulation with MSK (Minimum Shift Keying):
- Constant envelope (no zero crossings)
- ~2dB gain from full PA efficiency
- Continuous phase, cleaner spectrum

Usage:
    from msk2k_complete import MSK2KTransmitter, MSK2KReceiver
    
    # Transmit
    tx = MSK2KTransmitter(sample_rate=48000)
    audio = tx.generate_cq('G2NXX')
    
    # Receive
    rx = MSK2KReceiver(sample_rate=48000)
    result = rx.decode(audio, my_callsign='G2NXX')
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from scipy import signal as sp_signal
from scipy.io import wavfile
import time


# =============================================================================
# PART 1: SOURCE ENCODER
# From: psk2k_source_encoder.py (471 lines)
# =============================================================================

class PSK2kSourceEncoder:
    """
    PSK2k Source Encoder
    Encodes callsigns and messages into binary format
    """
    
    def __init__(self):
        """Initialize the source encoder"""
        # Base-37 alphabet for callsigns per Section 7.1
        # Index: 0=/, 1=A, 2=B, ..., 26=Z, 27=0, 28=1, ..., 36=9
        self.callsign_alphabet = "/ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        # Base-42 alphabet for text messages (includes punctuation)
        self.text_alphabet = " /ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-?"
        
        # Prime numbers for parity generation
        self.primes = {
            7: 3,   # 7 gives 3 parity bits
            13: 4,
            23: 5,
            29: 5,
            31: 5,
            59: 6,
            61: 6
        }
        
        # Parity selection by callsign length
        self.parity_selection = {
            3: [7, 13, 23, 29, 31, 59, 61],
            4: [7, 13, 29, 31, 59, 61],
            5: [7, 13, 29, 31, 61],
            6: [7, 13, 29, 31],
            7: [7, 29, 31],
            8: [7, 31],
            9: [7],
            10: []  # No parity for 10-character calls
        }
        
        # Length codes (last 4 or 1 bits of 54-bit callsign encoding)
        self.length_codes = {
            3: '0111',
            4: '1011',
            5: '0011',
            6: '1101',
            7: '0101',
            8: '1001',
            9: '0001',
            10: '0'
        }
    
    def encode_callsign(self, callsign: str) -> np.ndarray:
        """
        Encode a callsign to 54-bit binary representation
        
        Args:
            callsign: Amateur radio callsign (3-10 characters)
            
        Returns:
            54-bit numpy array
        """
        call = callsign.upper().strip()
        call_len = len(call)
        
        if call_len < 3 or call_len > 10:
            raise ValueError(f"Callsign length must be 3-10 characters, got {call_len}")
        
        # Convert callsign to base-37 number
        z = 0
        for char in call:
            if char not in self.callsign_alphabet:
                char = ' '
            digit = self.callsign_alphabet.index(char)
            z = z * 37 + digit
        
        # Determine binary representation length
        bit_lengths = {3: 16, 4: 21, 5: 27, 6: 32, 7: 37, 8: 42, 9: 47}
        
        if call_len == 10:
            # Special case: split into two parts
            call1 = call[:6]
            call2 = call[6:]
            
            z1 = 0
            for char in call1:
                if char not in self.callsign_alphabet:
                    char = ' '
                z1 = z1 * 37 + self.callsign_alphabet.index(char)
            
            z2 = 0
            for char in call2:
                if char not in self.callsign_alphabet:
                    char = ' '
                z2 = z2 * 37 + self.callsign_alphabet.index(char)
            
            # 32 bits + 21 bits + 1 length bit = 54 bits
            bits1 = self._int_to_bits(z1, 32)
            bits2 = self._int_to_bits(z2, 21)
            length_bits = [0]  # Length code for 10 chars
            
            return np.array(bits1 + bits2 + length_bits, dtype=int)
        
        # Normal case: encode as single number
        bit_len = bit_lengths[call_len]
        call_bits = self._int_to_bits(z, bit_len)
        
        # Generate parity bits
        parity_bits = []
        for prime in self.parity_selection[call_len]:
            remainder = z % prime
            num_bits = self.primes[prime]
            parity_bits.extend(self._int_to_bits(remainder, num_bits))
        
        # Add length code (4 bits for lengths 3-9)
        length_code = [int(b) for b in self.length_codes[call_len]]
        
        # Combine: callsign + parity + length = 54 bits
        # Length code goes at bits 50-53, parity fills 32-49
        # Total must be exactly 54 bits
        total_bits = call_bits + parity_bits
        
        # Pad parity to fill up to bit 50 (so length code is at 50-53)
        while len(total_bits) < 50:
            total_bits.append(0)
        
        # Truncate if somehow too long (shouldn't happen)
        total_bits = total_bits[:50]
        
        # Add length code at positions 50-53
        total_bits.extend(length_code)
        
        return np.array(total_bits[:54], dtype=int)
    
    def decode_callsign(self, bits: np.ndarray) -> str:
        """
        Decode 54-bit array back to callsign
        
        Args:
            bits: 54-bit array
            
        Returns:
            Callsign string
        """
        if len(bits) != 54:
            return "ERROR"
        
        # Determine length from last bits
        if bits[53] == 0:
            # 10-character callsign
            bits1 = bits[:32]
            bits2 = bits[32:53]
            
            z1 = self._bits_to_int(bits1)
            z2 = self._bits_to_int(bits2)
            
            call1 = self._base37_decode(z1, 6)
            call2 = self._base37_decode(z2, 4)
            
            return (call1 + call2).strip()
        
        # Determine length from 4-bit code
        length_code = ''.join(map(str, bits[50:54]))
        
        # Find matching length
        call_len = None
        for length, code in self.length_codes.items():
            if code == length_code:
                call_len = length
                break
        
        if call_len is None:
            return "ERROR"
        
        # Extract callsign bits
        bit_lengths = {3: 16, 4: 21, 5: 27, 6: 32, 7: 37, 8: 42, 9: 47}
        bit_len = bit_lengths[call_len]
        
        call_bits = bits[:bit_len]
        z = self._bits_to_int(call_bits)
        
        callsign = self._base37_decode(z, call_len)
        
        return callsign.strip()
    
    def generate_private_address(self, callsign: str) -> np.ndarray:
        """
        Generate 50-bit private address from callsign (not 49 as spec says)
        
        Args:
            callsign: Target callsign
            
        Returns:
            50-bit private address (based on DJ5HG's working implementation)
        """
        full_code = self.encode_callsign(callsign)
        
        # Check if 10-character callsign (length code = 0)
        if full_code[-1] == 0 and np.all(full_code[-4:] == 0):
            # Length 10: erase bits at indices 49:54, keep 0:49
            address = full_code[:49]
        else:
            # Other lengths: erase bits at indices 46:50
            # Keep bits 0:46 and 50:54 = 50 bits total
            address = np.concatenate([full_code[:46], full_code[50:]])
        
        return address
    
    def decode_private_address(self, addr_bits: np.ndarray) -> str:
        """
        Decode callsign from 49-bit private address.
        
        The private address is generate_private_address()[:49], which contains:
        - bits 0:46 of full encoding (callsign + partial parity)
        - bits 46:49 are from full[50:53] (first 3 bits of length code)
        
        To decode: reconstruct 54-bit encoding and decode.
        """
        if len(addr_bits) < 49:
            return "ERROR"
        
        # Reconstruct 54-bit encoding:
        # - bits 0:46 come from addr[0:46] 
        # - bits 46:50 were erased (set to 0)
        # - bits 50:53 come from addr[46:49]
        # - bit 53 we need to guess (usually 1 for 3-9 char calls)
        
        reconstructed = np.zeros(54, dtype=int)
        reconstructed[0:46] = addr_bits[0:46]
        # bits 46:50 stay 0 (erased parity - not needed for callsign)
        reconstructed[50:53] = addr_bits[46:49]
        reconstructed[53] = 1  # Assume not a 10-char call
        
        try:
            return self.decode_callsign(reconstructed)
        except:
            # Try with bit 53 = 0 (10-char callsign)
            reconstructed[53] = 0
            try:
                return self.decode_callsign(reconstructed)
            except:
                return "UNKNOWN"
        
        return address
    
    def _int_to_bits(self, value: int, num_bits: int) -> List[int]:
        """Convert integer to binary bit list (MSB first)"""
        bits = []
        for i in range(num_bits - 1, -1, -1):
            bits.append((value >> i) & 1)
        return bits
    
    def _bits_to_int(self, bits) -> int:
        """Convert binary bit list to integer"""
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value
    
    def _base37_decode(self, value: int, length: int) -> str:
        """Decode base-37 number to callsign"""
        chars = []
        for _ in range(length):
            chars.append(self.callsign_alphabet[value % 37])
            value //= 37
        return ''.join(reversed(chars))
    
    def generate_parity(self, source_bits: np.ndarray, r: int, num_bits: int) -> np.ndarray:
        """
        Generate parity bits using residual code
        
        Args:
            source_bits: Source bit array
            r: Modulo value (prime number)
            num_bits: Number of parity bits to generate
            
        Returns:
            Parity bit array
        """
        # Convert bits to integer
        value = self._bits_to_int(source_bits)
        
        # Calculate remainder
        remainder = value % r
        
        # Convert to binary
        return np.array(self._int_to_bits(remainder, num_bits), dtype=int)


# =============================================================================
# PART 2: CONVOLUTIONAL ENCODER
# From: psk2k_convolutional_encoder.py (215 lines)
# =============================================================================

class PSK2kConvolutionalEncoder:
    """
    Convolutional encoder for PSK2k
    Implements both Format 1 (rate 1/2) and Format 2 (rate 1/9)
    """
    
    def __init__(self):
        """Initialize with generator polynomials from specification"""
        
        # Format 1: Rate 1/2, constraint length 13
        self.format1_polynomials = [
            [1,1,0,1,1,0,1,0,1,0,0,0,1],  # 1101101010001
            [1,0,0,0,1,1,0,1,1,1,1,1,1]   # 1000110111111
        ]
        self.format1_constraint_length = 13
        
        # Format 2: Rate 1/9, constraint length 10
        self.format2_polynomials = [
            [1,1,1,1,0,0,1,0,0,1],  # 1111001001
            [1,0,1,0,1,1,1,1,0,1],  # 1010111101
            [1,1,0,1,1,0,0,1,1,1],  # 1101100111
            [1,1,0,1,0,1,0,1,1,1],  # 1101010111
            [1,1,1,1,0,0,1,0,0,1],  # 1111001001
            [1,0,1,0,1,1,1,1,0,1],  # 1010111101
            [1,1,0,1,1,0,0,1,1,1],  # 1101100111
            [1,1,1,0,1,1,1,0,0,1],  # 1110111001
            [1,0,1,0,0,1,1,0,1,1]   # 1010011011
        ]
        self.format2_constraint_length = 10
    
    def convolve_polynomial(self, info_bits: np.ndarray, polynomial: List[int]) -> np.ndarray:
        """
        Apply single polynomial to information bits
        
        Args:
            info_bits: Information bit array
            polynomial: Generator polynomial
            
        Returns:
            Encoded bit array
        """
        poly = np.array(polynomial)
        poly_len = len(poly)
        output_len = len(info_bits) + poly_len - 1
        
        # Create matrix where each '1' in info_bits is replaced by polynomial
        matrix = np.zeros((len(info_bits), output_len), dtype=int)
        
        for i, bit in enumerate(info_bits):
            if bit == 1:
                matrix[i, i:i+poly_len] = poly
        
        # XOR columns (sum mod 2)
        encoded = np.sum(matrix, axis=0) % 2
        
        return encoded
    
    def encode_format1(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Encode using Format 1 (rate 1/2, constraint length 13)
        71 information bits → 166 coded bits
        
        Args:
            info_bits: 71 information bits
            
        Returns:
            166 encoded bits
        """
        if len(info_bits) != 71:
            raise ValueError(f"Format 1 requires 71 info bits, got {len(info_bits)}")
        
        # Add tail bits for tail-ending
        tail_len = 2 * (self.format1_constraint_length - 1)
        info_with_tail = np.concatenate([info_bits, np.zeros(tail_len, dtype=int)])
        
        # Encode with both polynomials
        encoded1 = self.convolve_polynomial(info_with_tail, self.format1_polynomials[0])
        encoded2 = self.convolve_polynomial(info_with_tail, self.format1_polynomials[1])
        
        # Take first 83 bits from each (71 + c - 1 = 83)
        encoded1 = encoded1[:83]
        encoded2 = encoded2[:83]
        
        # Interleave: alternate bits from both polynomials
        encoded = np.zeros(166, dtype=int)
        encoded[0::2] = encoded1
        encoded[1::2] = encoded2
        
        return encoded
    
    def encode_format2(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Encode using Format 2 (rate 1/9, constraint length 10)
        18 information bits → 162 coded bits (tail-biting)
        
        Args:
            info_bits: 18 information bits
            
        Returns:
            162 encoded bits
        """
        if len(info_bits) != 18:
            raise ValueError(f"Format 2 requires 18 info bits, got {len(info_bits)}")
        
        # Tail-biting: wrap last c-1 bits to beginning
        tail_len = self.format2_constraint_length - 1
        info_wrapped = np.concatenate([info_bits[-tail_len:], info_bits])
        
        # Encode with all 9 polynomials
        encoded_streams = []
        for polynomial in self.format2_polynomials:
            encoded = self.convolve_polynomial(info_wrapped, polynomial)
            # Take middle 18 bits (skip wrapped tail)
            encoded_streams.append(encoded[tail_len:tail_len+18])
        
        # Interleave all 9 streams
        encoded = np.zeros(162, dtype=int)
        for i in range(18):
            for j in range(9):
                encoded[i*9 + j] = encoded_streams[j][i]
        
        return encoded


# =============================================================================
# PART 3: INTERLEAVERS
# From: psk2k_interleaver_correct.py and psk2k_interleaver_format2.py
# =============================================================================

# Format 1 interleaving table (258 positions)
FORMAT1_TABLE = [
    ('S',1), ('1',1), ('A',1), ('2',19), ('1',36), ('2',54), ('S',2), ('1',15), ('A',2), ('2',33),
    ('1',50), ('2',68), ('S',3), ('1',29), ('A',3), ('2',47), ('1',64), ('2',82), ('S',4), ('1',43),
    ('A',4), ('2',61), ('1',78), ('1',12), ('S',5), ('1',57), ('A',5), ('2',75), ('2',9), ('1',26),
    ('S',6), ('1',71), ('A',6), ('1',5), ('2',23), ('1',40), ('S',7), ('2',2), ('A',7), ('1',19),
    ('2',37), ('1',54), ('S',8), ('2',16), ('A',8), ('1',33), ('2',51), ('1',68), ('S',9), ('2',30),
    ('A',9), ('1',47), ('2',65), ('1',82), ('S',10), ('2',44), ('A',10), ('1',61), ('2',79), ('2',13),
    ('S',11), ('2',58), ('A',11), ('1',75), ('1',9), ('2',27), ('S',12), ('2',72), ('A',12), ('2',6),
    ('1',23), ('2',41), ('S',13), ('1',2), ('A',13), ('2',20), ('1',37), ('2',55), ('S',14), ('1',16),
    ('A',14), ('2',34), ('1',51), ('2',69), ('S',15), ('1',30), ('A',15), ('2',48), ('1',65), ('2',83),
    ('S',16), ('1',44), ('A',16), ('2',62), ('1',79), ('1',13), ('S',17), ('1',58), ('A',17), ('2',76),
    ('2',10), ('1',27), ('S',18), ('1',72), ('A',18), ('1',6), ('2',24), ('1',41), ('S',19), ('2',3),
    ('A',19), ('1',20), ('2',38), ('1',55), ('S',20), ('2',17), ('A',20), ('1',34), ('2',52), ('1',69),
    ('S',21), ('2',31), ('A',21), ('1',48), ('2',66), ('1',83), ('S',22), ('2',45), ('A',22), ('1',62),
    ('2',80), ('2',14), ('S',23), ('2',59), ('A',23), ('1',76), ('1',10), ('2',28), ('S',24), ('2',73),
    ('A',24), ('2',7), ('1',24), ('2',42), ('S',25), ('1',3), ('A',25), ('2',21), ('1',38), ('2',56),
    ('S',26), ('1',17), ('A',26), ('2',35), ('1',52), ('2',70), ('S',27), ('1',31), ('A',27), ('2',49),
    ('1',66), ('1',14), ('S',28), ('1',45), ('1',28), ('2',63), ('1',80), ('A',28), ('S',29), ('1',59),
    ('A',29), ('2',77), ('2',11), ('1',42), ('S',30), ('1',73), ('A',30), ('1',7), ('2',25), ('1',56),
    ('S',31), ('2',4), ('A',31), ('1',21), ('2',39), ('1',70), ('S',32), ('2',18), ('A',32), ('1',35),
    ('2',53), ('2',1), ('S',33), ('2',32), ('A',33), ('1',49), ('2',67), ('2',15), ('S',34), ('2',46),
    ('A',34), ('1',63), ('2',81), ('2',29), ('S',35), ('2',60), ('A',35), ('1',77), ('1',11), ('2',43),
    ('S',36), ('2',74), ('A',36), ('2',8), ('1',25), ('2',57), ('S',37), ('1',4), ('A',37), ('2',22),
    ('1',39), ('2',71), ('S',38), ('1',18), ('A',38), ('2',36), ('1',53), ('A',44), ('S',39), ('1',32),
    ('A',39), ('2',50), ('1',67), ('A',45), ('S',40), ('1',46), ('A',40), ('2',64), ('1',81), ('A',46),
    ('S',41), ('1',60), ('A',41), ('2',78), ('2',12), ('A',47), ('S',42), ('1',74), ('A',42), ('1',8),
    ('2',26), ('A',48), ('S',43), ('2',5), ('A',43), ('1',22), ('2',40), ('A',49),
]

# Format 2 interleaving table (258 positions)
FORMAT2_TABLE = [
    ('s',1), ('pa',1), ('a',1), ('pc',13), ('pe',8), ('ph',11), ('s',2), ('pa',10), ('a',2), ('pc',5),
    ('pe',17), ('ph',3), ('s',3), ('pa',2), ('a',3), ('pc',14), ('pe',9), ('ph',12), ('s',4), ('pa',11),
    ('a',4), ('pc',6), ('pe',18), ('ph',4), ('s',5), ('pa',3), ('a',5), ('pc',15), ('pf',1), ('ph',13),
    ('s',6), ('pa',12), ('a',6), ('pc',7), ('pf',10), ('ph',5), ('s',7), ('pa',4), ('a',7), ('pc',16),
    ('pf',2), ('ph',14), ('s',8), ('pa',13), ('a',8), ('pc',8), ('pf',11), ('ph',6), ('s',9), ('pa',5),
    ('a',9), ('pc',17), ('pf',3), ('ph',15), ('s',10), ('pa',14), ('a',10), ('pc',9), ('pf',12), ('ph',7),
    ('s',11), ('pa',6), ('a',11), ('pc',18), ('pf',4), ('ph',16), ('s',12), ('pa',15), ('a',12), ('pd',1),
    ('pf',13), ('ph',8), ('s',13), ('pa',7), ('a',13), ('pd',10), ('pf',5), ('ph',17), ('s',14), ('pa',16),
    ('a',14), ('pd',2), ('pf',14), ('ph',9), ('s',15), ('pa',8), ('a',15), ('pd',11), ('pf',6), ('ph',18),
    ('s',16), ('pa',17), ('a',16), ('pd',3), ('pf',15), ('pi',1), ('s',17), ('pa',9), ('a',17), ('pd',12),
    ('pf',7), ('pi',10), ('s',18), ('pa',18), ('a',18), ('pd',4), ('pf',16), ('pi',2), ('s',19), ('pb',1),
    ('a',19), ('pd',13), ('pf',8), ('pi',11), ('s',20), ('pb',10), ('a',20), ('pd',5), ('pf',17), ('pi',3),
    ('s',21), ('pb',2), ('a',21), ('pd',14), ('pf',9), ('pi',12), ('s',22), ('pb',11), ('a',22), ('pd',6),
    ('pf',18), ('pi',4), ('s',23), ('pb',3), ('a',23), ('pd',15), ('pg',1), ('pi',13), ('s',24), ('pb',12),
    ('a',24), ('pd',7), ('pg',10), ('pi',5), ('s',25), ('pb',4), ('a',25), ('pd',16), ('pg',2), ('pi',14),
    ('s',26), ('pb',13), ('a',26), ('pd',8), ('pg',11), ('pi',6), ('s',27), ('pb',5), ('a',27), ('pd',17),
    ('pg',3), ('pi',15), ('s',28), ('pb',14), ('a',28), ('pd',9), ('pg',12), ('pi',7), ('s',29), ('pb',6),
    ('a',29), ('pd',18), ('pg',4), ('pi',16), ('s',30), ('pb',15), ('a',30), ('pe',1), ('pg',13), ('pi',8),
    ('s',31), ('pb',7), ('a',31), ('pe',10), ('pg',5), ('pi',17), ('s',32), ('pb',16), ('a',32), ('pe',2),
    ('pg',14), ('pi',9), ('s',33), ('pb',8), ('a',33), ('pe',11), ('pg',6), ('pi',18), ('s',34), ('pb',17),
    ('a',34), ('pe',3), ('pg',15), ('_',0), ('s',35), ('pb',9), ('a',35), ('pe',12), ('pg',7), ('_',0),
    ('s',36), ('pb',18), ('a',36), ('pe',4), ('pg',16), ('_',0), ('s',37), ('pc',1), ('a',37), ('pe',13),
    ('pg',8), ('_',0), ('s',38), ('pc',10), ('a',38), ('pe',5), ('pg',17), ('a',44), ('s',39), ('pc',2),
    ('a',39), ('pe',14), ('pg',9), ('a',45), ('s',40), ('pc',11), ('a',40), ('pe',6), ('pg',18), ('a',46),
    ('s',41), ('pc',3), ('a',41), ('pe',15), ('ph',1), ('a',47), ('s',42), ('pc',12), ('a',42), ('pe',7),
    ('ph',10), ('a',48), ('s',43), ('pc',4), ('a',43), ('pe',16), ('ph',2), ('a',49),
]


def interleave_format1(sync_bits, addr_bits, poly1_bits, poly2_bits):
    """Interleave Format 1 packet"""
    if len(sync_bits) != 43 or len(addr_bits) != 49:
        raise ValueError("Invalid sync/addr lengths")
    if len(poly1_bits) != 83 or len(poly2_bits) != 83:
        raise ValueError("Invalid polynomial lengths")
    
    packet = np.zeros(258, dtype=int)
    
    for position, (type_code, index) in enumerate(FORMAT1_TABLE):
        idx = index - 1
        if type_code == 'S':
            packet[position] = sync_bits[idx]
        elif type_code == 'A':
            packet[position] = addr_bits[idx]
        elif type_code == '1':
            packet[position] = poly1_bits[idx]
        elif type_code == '2':
            packet[position] = poly2_bits[idx]
    
    return packet


def deinterleave_format1(packet):
    """Deinterleave Format 1 packet"""
    if len(packet) != 258:
        raise ValueError(f"Need 258-bit packet, got {len(packet)}")
    
    sync_bits = np.zeros(43, dtype=int)
    addr_bits = np.zeros(49, dtype=int)
    poly1_bits = np.zeros(83, dtype=int)
    poly2_bits = np.zeros(83, dtype=int)
    
    for position, (type_code, index) in enumerate(FORMAT1_TABLE):
        idx = index - 1
        if type_code == 'S':
            sync_bits[idx] = packet[position]
        elif type_code == 'A':
            addr_bits[idx] = packet[position]
        elif type_code == '1':
            poly1_bits[idx] = packet[position]
        elif type_code == '2':
            poly2_bits[idx] = packet[position]
    
    return sync_bits, addr_bits, poly1_bits, poly2_bits


def interleave_format2(sync_bits, addr_bits, poly_bits_dict):
    """Interleave Format 2 packet"""
    if len(sync_bits) != 43 or len(addr_bits) != 49:
        raise ValueError("Invalid sync/addr lengths")
    
    poly_names = ['Pa', 'Pb', 'Pc', 'Pd', 'Pe', 'Pf', 'Pg', 'Ph', 'Pi']
    for poly_name in poly_names:
        if poly_name not in poly_bits_dict or len(poly_bits_dict[poly_name]) != 18:
            raise ValueError(f"Invalid polynomial {poly_name}")
    
    packet = np.zeros(258, dtype=int)
    
    for position, (type_code, index) in enumerate(FORMAT2_TABLE):
        if type_code == '_':
            packet[position] = 0
            continue
        
        idx = index - 1
        type_code_upper = type_code.upper()
        
        if type_code_upper == 'S':
            packet[position] = sync_bits[idx]
        elif type_code_upper == 'A':
            packet[position] = addr_bits[idx]
        elif type_code_upper in [p.upper() for p in poly_names]:
            for poly_name in poly_names:
                if type_code_upper == poly_name.upper():
                    packet[position] = poly_bits_dict[poly_name][idx]
                    break
    
    return packet


def deinterleave_format2(packet):
    """Deinterleave Format 2 packet"""
    if len(packet) != 258:
        raise ValueError(f"Need 258-bit packet, got {len(packet)}")
    
    sync_bits = np.zeros(43, dtype=int)
    addr_bits = np.zeros(49, dtype=int)
    poly_dict = {name: np.zeros(18, dtype=int) for name in 
                 ['Pa', 'Pb', 'Pc', 'Pd', 'Pe', 'Pf', 'Pg', 'Ph', 'Pi']}
    
    for position, (type_code, index) in enumerate(FORMAT2_TABLE):
        if type_code == '_':
            continue
        
        idx = index - 1
        type_code_upper = type_code.upper()
        
        if type_code_upper == 'S':
            sync_bits[idx] = packet[position]
        elif type_code_upper == 'A':
            addr_bits[idx] = packet[position]
        elif type_code_upper in [p.upper() for p in poly_dict.keys()]:
            for poly_name in poly_dict.keys():
                if type_code_upper == poly_name.upper():
                    poly_dict[poly_name][idx] = packet[position]
                    break
    
    return sync_bits, addr_bits, poly_dict


# =============================================================================
# PART 4: MSK MODULATOR
# MSK (Minimum Shift Keying) - constant envelope, continuous phase
# =============================================================================

class MSK2KModulator:
    """MSK2K modulator at 2000 bits/s
    
    MSK is continuous-phase FSK with modulation index h=0.5
    - Frequency deviation = bit_rate / 4 = 500 Hz
    - Phase changes ±π/2 per symbol (±90°)
    - Constant envelope - no zero crossings, full PA efficiency
    
    Bit mapping:
    - Bit 1: frequency = carrier + 500 Hz (phase advances +90°/symbol)
    - Bit 0: frequency = carrier - 500 Hz (phase advances -90°/symbol)
    """
    
    def __init__(self, sample_rate: int = 48000):
        """Initialize MSK modulator"""
        self.sample_rate = sample_rate
        self.bit_rate = 2000
        self.carrier_freq = 1500.0  # Center frequency
        self.packet_bits = 258
        self.samples_per_bit = int(self.sample_rate / self.bit_rate)
        
        # MSK frequency deviation = bit_rate / 4
        self.freq_deviation = self.bit_rate / 4  # 500 Hz
    
    def modulate_msk(self, bits: np.ndarray) -> np.ndarray:
        """Generate MSK audio signal from bits.
        
        MSK is generated by accumulating phase continuously.
        Each bit causes phase to change by ±π/2 over the symbol period.
        
        Args:
            bits: Binary data (0s and 1s)
            
        Returns:
            Audio signal (constant envelope, continuous phase)
        """
        n_bits = len(bits)
        n_samples = n_bits * self.samples_per_bit
        
        # Build continuous phase
        phase = np.zeros(n_samples)
        current_phase = 0.0
        
        for i, bit in enumerate(bits):
            # Phase increment per symbol: +π/2 for bit 1, -π/2 for bit 0
            delta_phase = np.pi / 2 if bit else -np.pi / 2
            phase_rate = delta_phase / self.samples_per_bit
            
            for j in range(self.samples_per_bit):
                idx = i * self.samples_per_bit + j
                phase[idx] = current_phase + phase_rate * (j + 0.5)  # Sample at center
            
            current_phase += delta_phase
        
        # Generate signal at carrier frequency
        t = np.arange(n_samples) / self.sample_rate
        audio = np.cos(2 * np.pi * self.carrier_freq * t + phase)
        
        return audio
    
    def generate_packet_audio(self, bits: np.ndarray) -> np.ndarray:
        """Generate complete audio signal from packet bits"""
        if len(bits) != self.packet_bits:
            raise ValueError(f"Packet must be {self.packet_bits} bits, got {len(bits)}")
        
        audio = self.modulate_msk(bits)
        
        # Normalize (MSK is already constant envelope, but ensure ±1 range)
        audio = audio / np.max(np.abs(audio))
        
        return audio


# Keep PSK2kModulator as alias for backward compatibility
PSK2kModulator = MSK2KModulator


# =============================================================================
# PART 5: VITERBI DECODER  
# From: psk2k_viterbi_decoder.py (341 lines)
# =============================================================================

class PSK2kViterbiDecoder:
    """Viterbi decoder for PSK2k convolutional codes - fully vectorized"""
    
    def __init__(self):
        """Initialize with generator polynomials and precomputed tables"""
        self.format1_polynomials = [0b1101101010001, 0b1000110111111]
        self.format1_k = 13
        
        self.format2_polynomials = [
            0b1111100101, 0b1010111101, 0b1101100111, 0b1101010111,
            0b1111100101, 0b1010111101, 0b1101100111, 0b1110111001, 0b1010011011
        ]
        self.format2_k = 10
        
        # Precompute lookup tables for Format 1 (K=13, 4096 states, rate 1/2)
        self._init_format1_tables()
        
        # Precompute lookup tables for Format 2 (K=10, 512 states, rate 1/9)
        self._init_format2_tables()
    
    def _init_format1_tables(self):
        """
        Precompute REVERSE transition tables for vectorized ACS.
        
        For each state s, we need to know its two predecessors and what
        output bits they produce when transitioning to s.
        """
        K = self.format1_k
        num_states = 2 ** (K - 1)  # 4096
        rate = len(self.format1_polynomials)  # 2
        
        # For state s, predecessor with input=0 is: (s << 1) & mask
        # For state s, predecessor with input=1 is: ((s << 1) | 1) & mask
        # But we need the REVERSE: given next_state, find predecessors
        
        # Actually easier: for each state, the two NEXT states are:
        #   input=0: (0 << (K-2)) | (state >> 1) = state >> 1
        #   input=1: (1 << (K-2)) | (state >> 1) = (1 << (K-2)) | (state >> 1)
        
        # So predecessors of state s are:
        #   p0 = (s << 1) & (num_states - 1)      with input_bit = s >> (K-2)
        #   p1 = ((s << 1) | 1) & (num_states - 1) with input_bit = s >> (K-2)
        # Wait, that's not right either. Let me think...
        
        # State = K-1 bits of shift register (excluding current input)
        # next_state = (input_bit << (K-2)) | (state >> 1)
        # 
        # So if next_state = s, then:
        #   s = (input_bit << (K-2)) | (prev_state >> 1)
        #   s & ((1 << (K-2)) - 1) = prev_state >> 1
        #   prev_state = ((s & ((1 << (K-2)) - 1)) << 1) | lsb
        #   where lsb can be 0 or 1 (two predecessors)
        #   and input_bit = s >> (K-2)
        
        mask_low = (1 << (K - 2)) - 1  # Lower K-2 bits
        
        # Predecessors: prev0[s], prev1[s] - the two states that can transition to s
        self.f1_prev0 = np.zeros(num_states, dtype=np.uint16)
        self.f1_prev1 = np.zeros(num_states, dtype=np.uint16)
        # Input bit that caused the transition (same for both predecessors of s)
        self.f1_input_bit = np.zeros(num_states, dtype=np.uint8)
        # Output signs for soft metric: +1 or -1 for each polynomial
        # Shape: (num_states, 2, rate) - [state, which_predecessor, poly]
        self.f1_output_signs = np.zeros((num_states, 2, rate), dtype=np.float32)
        
        for s in range(num_states):
            input_bit = (s >> (K - 2)) & 1
            base = (s & mask_low) << 1
            p0 = base        # predecessor with LSB=0
            p1 = base | 1    # predecessor with LSB=1
            
            self.f1_prev0[s] = p0
            self.f1_prev1[s] = p1
            self.f1_input_bit[s] = input_bit
            
            # Compute output bits for transitions p0->s and p1->s
            for pi, prev in enumerate([p0, p1]):
                shift_reg = (input_bit << (K - 1)) | prev
                for poly_idx, poly in enumerate(self.format1_polynomials):
                    out_bit = (shift_reg & poly).bit_count() & 1
                    # Sign: +1 if output is 1, -1 if output is 0
                    self.f1_output_signs[s, pi, poly_idx] = 1.0 if out_bit else -1.0
    
    def _init_format2_tables(self):
        """Precompute tables for Format 2 (K=10, 512 states, rate 1/9)"""
        K = self.format2_k
        num_states = 2 ** (K - 1)  # 512
        rate = len(self.format2_polynomials)  # 9
        
        mask_low = (1 << (K - 2)) - 1
        
        self.f2_prev0 = np.zeros(num_states, dtype=np.uint16)
        self.f2_prev1 = np.zeros(num_states, dtype=np.uint16)
        self.f2_input_bit = np.zeros(num_states, dtype=np.uint8)
        self.f2_output_signs = np.zeros((num_states, 2, rate), dtype=np.float32)
        
        for s in range(num_states):
            input_bit = (s >> (K - 2)) & 1
            base = (s & mask_low) << 1
            p0 = base
            p1 = base | 1
            
            self.f2_prev0[s] = p0
            self.f2_prev1[s] = p1
            self.f2_input_bit[s] = input_bit
            
            for pi, prev in enumerate([p0, p1]):
                shift_reg = (input_bit << (K - 1)) | prev
                for poly_idx, poly in enumerate(self.format2_polynomials):
                    out_bit = (shift_reg & poly).bit_count() & 1
                    self.f2_output_signs[s, pi, poly_idx] = 1.0 if out_bit else -1.0
    
    def decode_format1(self, soft_bits: np.ndarray):
        """
        Decode Format 1 (rate 1/2, K=13)
        
        Returns:
            Tuple of (decoded_bits, path_metric) if called from accumulator
            Or just decoded_bits for backward compatibility
        """
        decoded, metric = self._viterbi_soft(soft_bits, self.format1_polynomials, 
                                  self.format1_k, num_info_bits=71)
        return decoded, metric
    
    def decode_format2(self, soft_bits: np.ndarray):
        """
        Decode Format 2 (rate 1/9, K=10, tail-biting)
        
        Uses efficient wrap-around Viterbi instead of trying all 512 start states.
        
        Returns:
            Tuple of (decoded_bits, path_metric)
        """
        K = self.format2_k
        num_states = 2 ** (K - 1)  # 512
        rate = len(self.format2_polynomials)  # 9
        num_steps = len(soft_bits) // rate  # 18
        
        # Precompute branch metrics for all states and input bits
        # This avoids recomputing inside the loop
        branch_metrics = np.zeros((num_steps, num_states, 2))
        next_states = np.zeros((num_states, 2), dtype=int)
        outputs_table = np.zeros((num_states, 2, rate), dtype=int)
        
        # Build transition and output tables
        for state in range(num_states):
            for input_bit in [0, 1]:
                next_states[state, input_bit] = (input_bit << (K - 2)) | (state >> 1)
                shift_reg = (input_bit << (K - 1)) | state
                for p, poly in enumerate(self.format2_polynomials):
                    outputs_table[state, input_bit, p] = (shift_reg & poly).bit_count() % 2
        
        # Precompute branch metrics for each step
        for step in range(num_steps):
            rx_syms = soft_bits[step*rate:(step+1)*rate]
            for state in range(num_states):
                for input_bit in [0, 1]:
                    outputs = outputs_table[state, input_bit]
                    # Euclidean distance: (rx - expected)^2
                    expected = 2.0 * outputs - 1.0  # Map 0->-1, 1->+1
                    branch_metrics[step, state, input_bit] = np.sum((rx_syms - expected) ** 2)
        
        # Wrap-around Viterbi: run forward twice through the trellis
        # First pass: start from all states equally, find best ending state
        # Second pass: start from best ending state, decode
        
        INF = 1e30
        
        # First pass - start all states at 0
        metrics = np.full(num_states, INF)
        metrics[:] = 0  # All states equally likely initially
        decisions = np.zeros((num_steps * 2, num_states, 2), dtype=int)  # (prev_state, input_bit)
        
        for wrap in range(2):  # Two passes around the trellis
            for step in range(num_steps):
                actual_step = wrap * num_steps + step
                new_metrics = np.full(num_states, INF)
                
                for state in range(num_states):
                    if metrics[state] >= INF:
                        continue
                    
                    for input_bit in [0, 1]:
                        ns = next_states[state, input_bit]
                        new_metric = metrics[state] + branch_metrics[step, state, input_bit]
                        
                        if new_metric < new_metrics[ns]:
                            new_metrics[ns] = new_metric
                            decisions[actual_step, ns, 0] = state
                            decisions[actual_step, ns, 1] = input_bit
                
                metrics = new_metrics
        
        # Find best final state
        best_state = np.argmin(metrics)
        best_metric = metrics[best_state]
        
        # Traceback through second pass only (steps num_steps to 2*num_steps-1)
        decoded = []
        state = best_state
        
        for step in range(2 * num_steps - 1, num_steps - 1, -1):
            prev_state = decisions[step, state, 0]
            input_bit = decisions[step, state, 1]
            decoded.append(input_bit)
            state = prev_state
        
        decoded.reverse()
        return np.array(decoded[:18], dtype=int), best_metric
    
    def _viterbi_tail_biting(self, rx_soft, polynomials, K, start_state):
        """Viterbi decode with tail-biting"""
        rate = len(polynomials)
        num_states = 2 ** (K - 1)
        num_steps = len(rx_soft) // rate
        
        metrics = [np.full(num_states, np.inf) for _ in range(num_steps + 1)]
        decisions = [[None] * num_states for _ in range(num_steps)]
        
        metrics[0][start_state] = 0
        
        for step in range(num_steps):
            rx_syms = rx_soft[step*rate:(step+1)*rate]
            
            for prev_state in range(num_states):
                if metrics[step][prev_state] == np.inf:
                    continue
                
                for input_bit in [0, 1]:
                    next_state = (input_bit << (K - 2)) | (prev_state >> 1)
                    shift_reg = (input_bit << (K - 1)) | prev_state
                    
                    outputs = [(shift_reg & poly).bit_count() % 2 for poly in polynomials]
                    branch_metric = sum((rx_syms[i] - (1.0 if outputs[i] else -1.0))**2 
                                      for i in range(rate))
                    
                    new_metric = metrics[step][prev_state] + branch_metric
                    
                    if new_metric < metrics[step+1][next_state]:
                        metrics[step+1][next_state] = new_metric
                        decisions[step][next_state] = (prev_state, input_bit)
        
        final_metric = metrics[num_steps][start_state]
        
        if final_metric == np.inf:
            return np.zeros(num_steps, dtype=int), np.inf
        
        decoded = []
        state = start_state
        
        for step in range(num_steps - 1, -1, -1):
            if decisions[step][state] is None:
                decoded.append(0)
                state = start_state
            else:
                prev_state, input_bit = decisions[step][state]
                decoded.append(input_bit)
                state = prev_state
        
        decoded.reverse()
        return np.array(decoded, dtype=int), final_metric
    
    def _viterbi_soft(self, rx_soft, polynomials, K, num_info_bits, tail_biting=False):
        """
        Fully vectorized Viterbi decoder.
        
        Uses reverse transition tables and numpy vectorization for the ACS step.
        The inner loop is just a few numpy operations over all states at once.
        
        Branch metric for soft bits:
          bm = -sum(rx_soft * expected_sign)
        where expected_sign is +1 for output bit 1, -1 for output bit 0.
        (Negative because we want "more positive = better match" → lower metric)
        """
        rate = len(polynomials)
        num_states = 2 ** (K - 1)
        num_steps = len(rx_soft) // rate
        
        # Select appropriate precomputed tables
        if K == 13 and rate == 2:
            prev0 = self.f1_prev0
            prev1 = self.f1_prev1
            output_signs = self.f1_output_signs
        elif K == 10 and rate == 9:
            prev0 = self.f2_prev0
            prev1 = self.f2_prev1
            output_signs = self.f2_output_signs
        else:
            return self._viterbi_soft_fallback(rx_soft, polynomials, K, num_info_bits, tail_biting)
        
        INF = np.float32(1e30)
        
        # Reshape rx_soft: (num_steps, rate)
        rx_matrix = rx_soft[:num_steps * rate].reshape(num_steps, rate).astype(np.float32)
        
        # Precompute branch metrics for all states and both predecessors
        # bm[t, s, pi] = branch metric for predecessor pi transitioning to state s at time t
        # Using dot product: bm = -sum(rx * sign)  (negative = lower is better)
        # 
        # output_signs shape: (num_states, 2, rate)
        # rx_matrix shape: (num_steps, rate)
        # Result shape: (num_steps, num_states, 2)
        bm = -np.einsum('tr,spr->tsp', rx_matrix, output_signs)
        
        # Metrics array - use float32 for speed
        metrics = np.full(num_states, INF, dtype=np.float32)
        
        # Backpointers: which predecessor was chosen (0 or 1)
        backptr = np.zeros((num_steps, num_states), dtype=np.uint8)
        
        # Initialize
        if tail_biting:
            metrics[:] = 0.0
        else:
            metrics[0] = 0.0
        
        # Forward pass - fully vectorized ACS
        for t in range(num_steps):
            # Candidate metrics from both predecessors
            # cand0[s] = metrics[prev0[s]] + bm[t, s, 0]
            # cand1[s] = metrics[prev1[s]] + bm[t, s, 1]
            cand0 = metrics[prev0] + bm[t, :, 0]
            cand1 = metrics[prev1] + bm[t, :, 1]
            
            # Select better path
            choose1 = cand1 < cand0
            metrics = np.where(choose1, cand1, cand0)
            backptr[t] = choose1.astype(np.uint8)
        
        # Find best final state
        if tail_biting:
            final_state = int(np.argmin(metrics))
        else:
            final_state = 0
        
        # Traceback
        decoded = np.zeros(num_steps, dtype=np.uint8)
        state = final_state
        
        for t in range(num_steps - 1, -1, -1):
            # Get input bit for this state (precomputed)
            if K == 13:
                decoded[t] = self.f1_input_bit[state]
            else:
                decoded[t] = self.f2_input_bit[state]
            
            # Go to predecessor
            if backptr[t, state]:
                state = prev1[state]
            else:
                state = prev0[state]
        
        return decoded[:num_info_bits].astype(int), metrics[final_state]
    
    def _viterbi_soft_fallback(self, rx_soft, polynomials, K, num_info_bits, tail_biting=False):
        """Fallback Viterbi decoder for non-standard configurations"""
        rate = len(polynomials)
        num_states = 2 ** (K - 1)
        num_steps = len(rx_soft) // rate
        
        metrics = [np.full(num_states, np.inf) for _ in range(num_steps + 1)]
        decisions = [[None] * num_states for _ in range(num_steps)]
        
        if tail_biting:
            metrics[0][:] = 0
        else:
            metrics[0][0] = 0
        
        for step in range(num_steps):
            rx_syms = rx_soft[step*rate:(step+1)*rate]
            
            for prev_state in range(num_states):
                if metrics[step][prev_state] == np.inf:
                    continue
                
                for input_bit in [0, 1]:
                    next_state = (input_bit << (K - 2)) | (prev_state >> 1)
                    shift_reg = (input_bit << (K - 1)) | prev_state
                    
                    outputs = [(shift_reg & poly).bit_count() % 2 for poly in polynomials]
                    branch_metric = sum((rx_syms[i] - (1.0 if outputs[i] else -1.0))**2 
                                      for i in range(rate))
                    
                    new_metric = metrics[step][prev_state] + branch_metric
                    
                    if new_metric < metrics[step+1][next_state]:
                        metrics[step+1][next_state] = new_metric
                        decisions[step][next_state] = (prev_state, input_bit)
        
        final_state = np.argmin(metrics[num_steps]) if tail_biting else 0
        
        decoded = []
        state = final_state
        
        for step in range(num_steps - 1, -1, -1):
            if decisions[step][state] is None:
                decoded.append(0)
                state = 0
            else:
                prev_state, input_bit = decisions[step][state]
                decoded.append(input_bit)
                state = prev_state
        
        decoded.reverse()
        final_metric = metrics[num_steps][final_state]
        return np.array(decoded[:num_info_bits], dtype=int), final_metric


# =============================================================================
# PART 6: MSK2K RECEIVER
# MSK demodulation with differential phase detection
# =============================================================================

class MSK2KReceiver:
    """Complete MSK2K receiver with coherent MSK demodulation"""
    
    def __init__(self, sample_rate: int = 48000):
        """Initialize MSK receiver"""
        self.sample_rate = sample_rate
        self.carrier_freq = 1500.0  # Must match modulator
        self.bit_rate = 2000
        self.samples_per_bit = int(sample_rate / self.bit_rate)
        
        self.viterbi = PSK2kViterbiDecoder()
        self.source_decoder = PSK2kSourceEncoder()
        
        # Sync pattern (same as PSK2K - Hadamard-43)
        self.sync_pattern = np.array([
            0,1,0,0,1,0,1,0,0,1,1,1,0,1,1,1,1,1,0,0,
            0,1,0,1,1,1,0,0,0,0,0,1,0,0,0,1,1,0,1,0,1,1,0
        ])
        
        # PSK2K uses 3 shifted patterns for jigsaw assembly
        # CRITICAL: Store ACTUAL rotation amounts, not just labels
        self.sync_shifts = [0, 14, 29]  # These are just labels from spec
        self.sync_rotations = [0, 29, 14]  # These are the ACTUAL rotations used
        self.shifted_sync_patterns = []
        
        # Shift 0: original pattern (rotation 0)
        self.shifted_sync_patterns.append(self.sync_pattern.copy())
        
        # Shift 14 LABEL: Actually rotated LEFT by 29
        pattern_14 = np.concatenate([self.sync_pattern[29:], self.sync_pattern[:29]])
        self.shifted_sync_patterns.append(pattern_14)
        
        # Shift 29 LABEL: Actually rotated LEFT by 14
        pattern_29 = np.concatenate([self.sync_pattern[14:], self.sync_pattern[:14]])
        self.shifted_sync_patterns.append(pattern_29)
        
        # General address pattern
        general_pattern = "1101010000001111100110011011011011011011011011011"
        self.general_address = np.array([int(b) for b in general_pattern], dtype=int)
        
        # Tunable sync correlation threshold (depends on soft symbol scaling)
        # Lower = more sensitive (better at low SNR, more false positives)
        # Higher = stricter (better rejection, may miss weak signals)
        # Based on statistical testing with MSK2K:
        #   +5dB: mean=0.67, +3dB: mean=0.30, +1dB: mean=0.18, 0dB: mean=0.17
        # Threshold of 0.18 allows accumulation down to +1dB with ~40% detection rate
        # Tune using debug_sync=True to see correlation values for your setup
        self.sync_corr_threshold = 0.18
        
        # Debug mode: log sync attempts to help tune threshold
        self.debug_sync = False  # Set to True to see correlation values
        
        self.my_callsign = None
        self.partner_callsign = None
    
    def decode(self, audio: np.ndarray, my_callsign: Optional[str] = None,
               partner_callsign: Optional[str] = None) -> Dict:
        """
        Decode MSK2K audio (wrapper for compatibility with audio manager).
        
        Args:
            audio: Audio samples (numpy array)
            my_callsign: Receiving station callsign (for private address verification)
            partner_callsign: Expected transmitting station (for Format 2 display)
            
        Returns:
            Dict with decode result
        """
        # Set callsigns for this decode
        if my_callsign:
            self.my_callsign = my_callsign
        if partner_callsign:
            self.partner_callsign = partner_callsign
            
        return self._decode_signal(audio)
    
    def decode_wav(self, wav_file: str, my_callsign: Optional[str] = None,
                   partner_callsign: Optional[str] = None) -> Dict:
        """
        Decode PSK2k signal from WAV file
        
        Args:
            wav_file: Path to WAV file
            my_callsign: Receiver's callsign
            partner_callsign: QSO partner's callsign (for Format 2)
            
        Returns:
            Dictionary with decode results
        """
        # Load WAV
        sample_rate, audio = wavfile.read(wav_file)
        
        # Convert to float
        if audio.dtype == np.int16:
            audio = audio.astype(float) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(float) / 2147483648.0
        elif audio.dtype == np.float32:
            audio = audio.astype(float)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            from scipy.signal import resample
            num_samples = int(len(audio) * self.sample_rate / sample_rate)
            audio = resample(audio, num_samples)
        
        self.my_callsign = my_callsign
        self.partner_callsign = partner_callsign
        
        return self._decode_signal(audio)
    
    def decode_hard_bits(self, hard_bits: np.ndarray) -> Dict:
        """
        Decode pre-assembled hard bits from fragment accumulation
        
        This is used by the JavaScript UI after combining multiple
        weak meteor scatter pings into one averaged packet.
        
        Args:
            hard_bits: 258-bit array (0 or 1) from accumulated fragments
            
        Returns:
            Dictionary with decode results
        """
        if len(hard_bits) != 258:
            return {
                'success': False,
                'error': f'Expected 258 bits, got {len(hard_bits)}'
            }
        
        # Convert hard bits to soft bits for decoder
        # 0 → -1.0, 1 → +1.0
        soft_bits = 2.0 * np.array(hard_bits, dtype=float) - 1.0
        
        # Try Format 1
        result = self._try_decode_format1(soft_bits)
        if result['success']:
            result['source'] = 'assembled_hard_bits'
            return result
        
        # Try Format 2
        result = self._try_decode_format2(soft_bits)
        if result['success']:
            result['source'] = 'assembled_hard_bits'
            return result
        
        return {
            'success': False,
            'error': 'Could not decode assembled bits',
            'source': 'assembled_hard_bits'
        }
    
    def _decode_signal(self, audio: np.ndarray) -> Dict:
        """Decode MSK2K signal from audio samples.
        
        Uses intelligent multi-pass approach:
        1. Short signals (<0.3s): Quick single-packet decode
        2. Long signals (≥0.3s): Fast chunk scan first, then accumulator if needed
        
        Thresholds:
        - LONG_AUDIO_THRESHOLD = 0.30s (enables accumulation for ≥2 packets)
        - FAST_SCAN_CHUNK = 0.39s (≈3 packets, provides context for sync)
        - FAST_SCAN_STEP = 0.10s (sliding overlap)
        """
        duration = len(audio) / self.sample_rate
        
        # Thresholds
        LONG_AUDIO_THRESHOLD = 0.30  # Enable accumulation for ≥2 packets
        FAST_SCAN_CHUNK = 0.39       # ~3 packets for robust sync detection
        FAST_SCAN_STEP = 0.10        # Sliding window overlap
        
        # For long signals, do fast chunk scan first, then accumulation if needed
        if duration >= LONG_AUDIO_THRESHOLD:
            baseband = self._demodulate(audio)
            
            # Fast scan: Adaptive chunk size (don't exceed audio duration)
            chunk_duration = min(FAST_SCAN_CHUNK, duration)
            chunk_bits = int(chunk_duration * 2000)  # Convert to bits at 2000 baud
            step_bits = int(FAST_SCAN_STEP * 2000)
            packet_bits = 258
            
            # Ensure chunk is at least 1 packet
            chunk_bits = max(chunk_bits, packet_bits)
            
            for chunk_start in range(0, len(baseband) - packet_bits, step_bits):
                chunk_end = min(chunk_start + chunk_bits, len(baseband))
                chunk = baseband[chunk_start:chunk_end]
                
                # Quick sync check
                sync_result = self._find_sync(chunk)
                
                # Two-tier threshold:
                # >= 30: Try decode (marginal signals, Format 2 especially)
                # This matches the short-signal path threshold
                if sync_result['sync_bits'] >= 30:
                    packet_soft = self._extract_packet(chunk, sync_result)
                    if packet_soft is not None:
                        # Try Format 1 first
                        result = self._try_decode_format1(packet_soft)
                        if result['success']:
                            result['sync_bits'] = f"{sync_result['sync_bits']}/43"
                            result['sync_correlation'] = sync_result['correlation']
                            result['method'] = 'fast-scan'
                            return result
                        
                        # Try Format 2 if Format 1 failed
                        result = self._try_decode_format2(packet_soft)
                        if result['success']:
                            result['sync_bits'] = f"{sync_result['sync_bits']}/43"
                            result['sync_correlation'] = sync_result['correlation']
                            result['method'] = 'fast-scan'
                            return result
                        
                        # If sync >= 40 but decode failed, don't keep scanning
                        # (strong sync but bad decode = noise/interference)
                        # If sync 30-39 and decode failed, continue scanning
                        # (might find better packet elsewhere)
            
            # Fast scan didn't find anything - use accumulator for weak/fragmented signals
            result = self._decode_with_accumulation_baseband(baseband)
            return result
        
        # Short signal (<0.3s) - do quick single-packet decode
        baseband = self._demodulate(audio)
        sync_result = self._find_sync(baseband)
        
        # For single decode, try if we have reasonable sync (30+ sync bits)
        # Soft-decision Viterbi can handle marginal signals
        if sync_result['found'] and sync_result['sync_bits'] >= 30:
            packet_soft = self._extract_packet(baseband, sync_result)
            
            if packet_soft is not None:
                # Check address using SOFT correlation
                from msk2k_complete import deinterleave_format1_soft
                _, addr_soft, _, _ = deinterleave_format1_soft(packet_soft)
                
                general_addr_signed = (self.general_address * 2 - 1).astype(np.float32)
                addr_norm = np.linalg.norm(addr_soft)
                if addr_norm > 0:
                    soft_correlation = np.dot(addr_soft, general_addr_signed) / addr_norm / np.sqrt(len(addr_soft))
                else:
                    soft_correlation = 0.0
                
                is_general_addr = soft_correlation >= 0.15
                
                # Try Format 1 first
                if not is_general_addr:
                    result = self._try_decode_format1(packet_soft)
                    if result['success']:
                        result['sync_bits'] = f"{sync_result['sync_bits']}/43"
                        result['sync_correlation'] = sync_result['correlation']
                        result['method'] = 'single-packet'
                        return result
                    
                    result = self._try_decode_format2(packet_soft)
                    if result['success']:
                        result['sync_bits'] = f"{sync_result['sync_bits']}/43"
                        result['sync_correlation'] = sync_result['correlation']
                        result['method'] = 'single-packet'
                        return result
                else:
                    result = self._try_decode_format1(packet_soft)
                    if result['success']:
                        result['sync_bits'] = f"{sync_result['sync_bits']}/43"
                        result['sync_correlation'] = sync_result['correlation']
                        result['method'] = 'single-packet'
                        return result
                    
                    result = self._try_decode_format2(packet_soft)
                    if result['success']:
                        result['sync_bits'] = f"{sync_result['sync_bits']}/43"
                        result['sync_correlation'] = sync_result['correlation']
                        result['method'] = 'single-packet'
                        return result
        
        # Short packet that failed to decode
        return {
            'success': False,
            'error': f'Sync failed or decode failed (sync={sync_result["sync_bits"]}/43)',
            'sync_bits': f'{sync_result["sync_bits"]}/43',
            'sync_correlation': sync_result.get('correlation', 0)
        }
    
    def _decode_with_accumulation_baseband(self, baseband_full: np.ndarray) -> Dict:
        """Decode longer signal using multi-ping accumulation.
        
        OPTIMIZED: Works on already-demodulated baseband to avoid double FFT.
        Receives baseband soft bits from _decode_signal.
        """
        acc = PSK2kAccumulator(sample_rate=self.sample_rate)
        acc.reset()
        
        # baseband_full is already demodulated soft bits - no need to demod again!
        packet_bits = 258  # 258 bits per packet
        step_bits = 100  # Step size for sliding window
        
        # Need large enough chunks to allow for shift correction
        # Maximum correction is 6*29 = 174 bits backward
        # Use 2*packet + max_correction for safety
        max_correction = 174
        chunk_size = packet_bits * 2 + max_correction  # 690 bits
        
        # First pass: collect all sync results to find threshold
        all_results = []
        for start in range(0, len(baseband_full) - chunk_size, step_bits):
            chunk = baseband_full[start:start + chunk_size]
            
            sync_result = self._find_sync(chunk)
            all_results.append((start, sync_result))
        
        # CRITICAL: Use correlation-first for accumulation (not sync_bits)
        # This is the KEY to meteor scatter jigsaw assembly
        # Accept fragments with correlation >= threshold, even if sync_bits is low
        # A partial ping might have only 15-20 sync bits but still be valid
        corr_threshold = getattr(self, 'sync_corr_threshold', 0.18)
        
        # Count candidate pings for diagnostics
        pings_above_corr = sum(1 for r in all_results if r[1]['correlation'] >= corr_threshold)
        best_sync = max(r[1]['sync_bits'] for r in all_results) if all_results else 0
        best_corr = max(r[1]['correlation'] for r in all_results) if all_results else 0
        
        # PHASE CLUSTERING: First pass to extract candidates and compute phases
        candidates = []  # (start, sync_result, true_start, phase, canonical_corr, packet_soft)
        
        SYNC_POS = np.arange(0, 258, 6)[:43]
        canonical_sync = (self.sync_pattern * 2 - 1).astype(np.float32)
        canon_norm = np.linalg.norm(canonical_sync)
        
        for start, sync_result in all_results:
            if sync_result['correlation'] >= corr_threshold:
                chunk_rel_pos = sync_result['position']
                rotation = sync_result.get('sync_rotation', 0)
                correction = 6 * rotation
                true_start_in_baseband = start + chunk_rel_pos - correction
                
                if true_start_in_baseband < 0 or true_start_in_baseband + 258 > len(baseband_full):
                    continue
                
                # Extract and validate
                packet_soft = baseband_full[true_start_in_baseband:true_start_in_baseband + 258]
                polarity = sync_result.get('polarity', 1)
                packet_soft = packet_soft * polarity
                
                # CRITICAL FOR MSK: Extract magnitude for this packet
                # Magnitude indicates signal presence (high during pings, low in gaps)
                if hasattr(self, '_last_sym_mag') and self._last_sym_mag is not None:
                    if true_start_in_baseband + 258 <= len(self._last_sym_mag):
                        packet_mag = self._last_sym_mag[true_start_in_baseband:true_start_in_baseband + 258]
                    else:
                        packet_mag = None
                else:
                    packet_mag = None
                
                # Canonical correlation check
                sync_soft = packet_soft[SYNC_POS]
                sync_norm = np.linalg.norm(sync_soft)
                
                if sync_norm > 1e-12:
                    canonical_corr = np.dot(sync_soft, canonical_sync) / (sync_norm * canon_norm)
                else:
                    canonical_corr = 0.0
                
                canonical_threshold = getattr(self, 'canonical_corr_threshold', 0.30)
                
                if canonical_corr < canonical_threshold:
                    continue
                
                # Compute phase (mod 258 for packet alignment)
                phase = true_start_in_baseband % 258
                
                candidates.append({
                    'true_start': true_start_in_baseband,
                    'phase': phase,
                    'canonical_corr': canonical_corr,
                    'packet_soft': packet_soft,
                    'packet_mag': packet_mag,  # Store magnitude for accumulation
                    'weight': canonical_corr ** 2
                })
        
        # PHASE CLUSTERING: Find dominant phase bin
        if len(candidates) > 0:
            phases = np.array([c['phase'] for c in candidates])
            weights = np.array([c['weight'] for c in candidates])
            
            # Bin phases with tolerance ±6 bits
            phase_bins = {}
            for i, phase in enumerate(phases):
                # Find if this phase belongs to an existing bin
                found_bin = False
                for bin_center in phase_bins.keys():
                    if abs(phase - bin_center) <= 6:
                        phase_bins[bin_center]['indices'].append(i)
                        phase_bins[bin_center]['total_weight'] += weights[i]
                        found_bin = True
                        break
                
                if not found_bin:
                    phase_bins[phase] = {'indices': [i], 'total_weight': weights[i]}
            
            # Select dominant bin (highest total weight)
            if phase_bins:
                total_weight = sum(b['total_weight'] for b in phase_bins.values())
                dominant_center = max(phase_bins.keys(), key=lambda k: phase_bins[k]['total_weight'])
                dominant_indices = phase_bins[dominant_center]['indices']
                dominant_weight = phase_bins[dominant_center]['total_weight']
                
                # Diagnostic: Phase-bin dominance ratio (should be > 0.6)
                dominance_ratio = dominant_weight / total_weight if total_weight > 0 else 0
                
                # Diagnostic counters
                valid_bits_per_fragment = []
                canonical_corrs = []
                
                # Only accumulate candidates from dominant phase bin
                for idx in dominant_indices:
                    cand = candidates[idx]
                    packet_soft = cand['packet_soft']
                    packet_mag = cand['packet_mag']
                    weight = cand['weight']
                    
                    canonical_corrs.append(cand['canonical_corr'])
                    
                    # CRITICAL FOR MSK: Use magnitude for BOTH valid_mask and confidence
                    if packet_mag is not None:
                        # Magnitude-based valid mask (signal present/absent)
                        # Use Q25 * 0.8 but add floor to prevent mask flooding in quiet periods
                        q25 = np.quantile(packet_mag, 0.25)
                        q10 = np.quantile(packet_mag, 0.10)
                        
                        # Adaptive threshold with noise floor protection
                        mag_threshold = max(q25 * 0.8, q10 * 1.2, 0.05)
                        valid_mask = packet_mag >= mag_threshold
                        
                        # Diagnostic: Track valid bit fraction
                        valid_bits_per_fragment.append(np.sum(valid_mask))
                        
                        # Weighting: Use magnitude-squared for SNR-like behavior
                        # Squared weighting accelerates convergence on strong pings
                        conf_override = packet_mag ** 2
                        
                        acc.accumulate_soft_packet(packet_soft, weight, 
                                                  valid_mask=valid_mask,
                                                  conf_override=conf_override)
                    else:
                        # Fallback if magnitude not available
                        acc.accumulate_soft_packet(packet_soft, weight)
                
                # Store diagnostics for debugging
                acc._last_dominance_ratio = dominance_ratio
                acc._last_valid_bits_mean = np.mean(valid_bits_per_fragment) if valid_bits_per_fragment else 0
                acc._last_canonical_corr_mean = np.mean(canonical_corrs) if canonical_corrs else 0
        
        # Try accumulated decode
        if acc.num_pings > 0:
            result = acc.decode_accumulated(my_callsign=self.my_callsign,
                                           partner_callsign=self.partner_callsign)
            if result['success']:
                result['sync_bits'] = f"{acc.num_pings} pings accumulated"
                result['method'] = f'accumulator ({acc.num_pings} pings)'
                
                # Add health check diagnostics
                if hasattr(acc, '_last_dominance_ratio'):
                    result['phase_dominance'] = acc._last_dominance_ratio
                    result['avg_valid_bits'] = acc._last_valid_bits_mean
                    result['avg_canonical_corr'] = acc._last_canonical_corr_mean
                
                return result
            else:
                # Accumulated but decode failed - include diagnostics
                error_result = {
                    'success': False,
                    'error': f'Accumulated {acc.num_pings} pings but decode failed',
                    'sync_bits': f"{acc.num_pings} pings, decode failed",
                    'num_pings': acc.num_pings,
                    'best_sync': best_sync,
                    'correlation_threshold': corr_threshold
                }
                
                # Add diagnostics for debugging why it failed
                if hasattr(acc, '_last_dominance_ratio'):
                    error_result['phase_dominance'] = acc._last_dominance_ratio
                    error_result['avg_valid_bits'] = acc._last_valid_bits_mean
                    error_result['avg_canonical_corr'] = acc._last_canonical_corr_mean
                
                return error_result
        
        # No pings met both sync AND correlation thresholds
        if all_results:
            return {
                'success': False,
                'error': f'Found candidates but none passed validation',
                'sync_bits': f"best={best_sync}/43, {pings_above_corr} candidates",
                'best_sync_bits': best_sync,
                'pings_candidates': pings_above_corr
            }
        
        return {'success': False, 'error': 'No signal found', 'sync_bits': '0/43'}
    
    def _demodulate(self, audio: np.ndarray) -> np.ndarray:
        """Demodulate MSK signal using differential phase detection.
        
        Optimized approach with improved phase tracking:
        1. Create analytic signal via FFT
        2. Mix to baseband with frequency correction
        3. Phase estimation using sync correlation
        4. Differential phase detection for soft symbols
        
        Returns soft bit values for Viterbi decoder.
        """
        n = len(audio)
        sps = self.samples_per_bit
        
        # 1. Create analytic signal via FFT (one-sided spectrum)
        spectrum = np.fft.fft(audio)
        spectrum[n//2+1:] = 0  # Zero negative frequencies
        spectrum[1:n//2] *= 2   # Double positive frequencies
        analytic = np.fft.ifft(spectrum)
        
        # 2. Mix to baseband
        t = np.arange(n) / self.sample_rate
        lo = np.exp(-1j * 2 * np.pi * self.carrier_freq * t)
        baseband = analytic * lo
        
        # 3. Frequency offset estimation via squared spectrum
        if n > sps * 50:  # Only for longer signals
            squared = baseband ** 2
            sq_spectrum = np.fft.fft(squared)
            sq_freqs = np.fft.fftfreq(n, 1/self.sample_rate)
            
            search_mask = (np.abs(sq_freqs) > 800) & (np.abs(sq_freqs) < 1200)
            if np.any(search_mask):
                sq_power = np.abs(sq_spectrum)
                sq_power[~search_mask] = 0
                peak_idx = np.argmax(sq_power)
                peak_freq = sq_freqs[peak_idx]
                expected = 1000.0 if peak_freq > 0 else -1000.0
                freq_offset = (peak_freq - expected) / 2
                
                if abs(freq_offset) > 2:
                    correction = np.exp(-1j * 2 * np.pi * freq_offset * t)
                    baseband = baseband * correction
        
        # 4. Extract phase
        phase = np.angle(baseband)
        phase_unwrapped = np.unwrap(phase)
        
        # 5. Differential phase detection with symbol magnitude
        # CRITICAL FOR MSK: Magnitude indicates signal presence (envelope)
        # Unlike PSK, differential phase magnitude is large even in noise
        n_symbols = n // sps
        soft_bits = np.zeros(n_symbols)
        sym_mag = np.zeros(n_symbols)
        
        for i in range(n_symbols):
            start = i * sps
            end = start + sps
            if end >= n:
                break
            
            # Symbol magnitude - CRITICAL for MSK valid-bit masking
            # This captures signal presence/absence (envelope detection)
            # Using mean(abs(analytic_baseband)) over symbol interval
            sym_mag[i] = np.mean(np.abs(baseband[start:end]))
            
            # Differential phase for soft decision
            # Average over a few samples to reduce noise
            avg_window = max(1, sps // 4)
            
            start_phase = np.mean(phase_unwrapped[start:start+avg_window])
            end_phase = np.mean(phase_unwrapped[end-avg_window:end])
            
            delta_phase = end_phase - start_phase
            
            # Normalize: +π/2 → +1, -π/2 → -1
            soft_bits[i] = delta_phase / (np.pi / 2)
        
        # Store magnitude for accumulator to use as confidence metric
        # INVARIANT: len(sym_mag) == len(soft_bits) always
        self._last_sym_mag = sym_mag
        
        return soft_bits
    
    def _find_sync(self, soft_bits: np.ndarray) -> Dict:
        """
        Find sync pattern in soft bit stream using correlation-first + 3 shifted patterns.
        Returns best candidate with shift (0/14/29) and polarity (+1 / -1).
        
        OPTIMIZED with:
        - Stride slicing (samples[start:start+253:6]) instead of fancy indexing
        - All-phases coarse scan: Check all 6 phase offsets (0-5) with stride=6
        - Refine around best: ±5 samples with step=1
        
        This gives ~6× speedup while guaranteeing we don't miss any packet position.
        """
        samples = np.asarray(soft_bits).flatten()
        
        # Sync comb parameters
        SYNC_LEN = 43
        STEP = 6
        SYNC_SPAN = STEP * (SYNC_LEN - 1)  # 252
        
        best = {
            "correlation": -1.0,
            "position": -1,
            "sync_bits": 0,
            "polarity": 1,
            "sync_shift": 0,
        }
        
        # Precompute bipolar versions of shifted sync patterns
        shifted_sync_soft = [
            2.0 * pat.astype(float) - 1.0
            for pat in self.shifted_sync_patterns
        ]
        sync_norms = [np.sqrt(np.sum(p**2)) + 1e-12 for p in shifted_sync_soft]
        
        max_start = len(samples) - (SYNC_SPAN + 1)
        if max_start < 0:
            return {
                "found": False,
                "correlation": -1.0,
                "position": -1,
                "sync_bits": 0,
                "polarity": 1,
                "sync_shift": 0,
                "sync_rotation": 0,
            }
        
        # STAGE 1: Coarse search with stride=6, checking all 6 phase offsets
        # This ensures we don't miss packets at odd positions
        # For each phase offset (0-5), scan with stride=6
        for phase_offset in range(6):
            for packet_start in range(phase_offset, max_start + 1, 6):
                # OPTIMIZED: Stride slicing instead of fancy indexing
                window = samples[packet_start : packet_start + SYNC_SPAN + 1 : STEP]
                
                if len(window) != SYNC_LEN:
                    continue  # Skip this position, continue scanning
                
                # Precompute norm once for both polarities
                window_norm = float(np.sqrt(np.dot(window, window)) + 1e-12)
                
                # Try both polarities (phase ambiguity)
                for polarity in (1, -1):
                    test_window = window * polarity
                    tw_norm = window_norm
                    
                    # Evaluate all three shifted patterns
                    for shift_idx, sync_soft in enumerate(shifted_sync_soft):
                        corr = float(np.dot(test_window, sync_soft) / (tw_norm * sync_norms[shift_idx]))
                        
                        if corr > best["correlation"]:
                            # Only compute hard bits for new best
                            hard_bits = (test_window > 0).astype(int)
                            sync_bits = int(np.sum(hard_bits == self.shifted_sync_patterns[shift_idx]))
                            
                            best.update({
                                "correlation": corr,
                                "position": packet_start,
                                "sync_bits": sync_bits,
                                "polarity": polarity,
                                "sync_shift": self.sync_shifts[shift_idx],
                                "sync_rotation": self.sync_rotations[shift_idx],
                            })
        
        # STAGE 2: Refine search around best position (±5 samples with step=1)
        # This finds the exact peak within the 6-sample stride
        if best["position"] >= 0:
            best_coarse = best["position"]
            refine_start = max(0, best_coarse - 5)
            refine_end = min(max_start, best_coarse + 5)
            
            for packet_start in range(refine_start, refine_end + 1):
                # Use stride slicing
                window = samples[packet_start : packet_start + SYNC_SPAN + 1 : STEP]
                
                if len(window) != SYNC_LEN:
                    continue
                
                window_norm = float(np.sqrt(np.dot(window, window)) + 1e-12)
                
                for polarity in (1, -1):
                    test_window = window * polarity
                    tw_norm = window_norm
                    
                    for shift_idx, sync_soft in enumerate(shifted_sync_soft):
                        corr = float(np.dot(test_window, sync_soft) / (tw_norm * sync_norms[shift_idx]))
                        
                        if corr > best["correlation"]:
                            hard_bits = (test_window > 0).astype(int)
                            sync_bits = int(np.sum(hard_bits == self.shifted_sync_patterns[shift_idx]))
                            
                            best.update({
                                "correlation": corr,
                                "position": packet_start,
                                "sync_bits": sync_bits,
                                "polarity": polarity,
                                "sync_shift": self.sync_shifts[shift_idx],
                                "sync_rotation": self.sync_rotations[shift_idx],
                            })
        
        # IMPORTANT: decide "found" primarily by correlation, not sync_bits.
        corr_thresh = getattr(self, "sync_corr_threshold", 0.18)
        
        # Debug logging
        if getattr(self, "debug_sync", False) and best["position"] >= 0:
            print(f"[SYNC DEBUG] Best correlation: {best['correlation']:.3f}, "
                  f"threshold: {corr_thresh:.3f}, "
                  f"shift: {best['sync_shift']}, "
                  f"sync_bits: {best['sync_bits']}/43, "
                  f"found: {best['correlation'] >= corr_thresh}")
        
        # NOTE: We return raw position and rotation
        # - _extract_packet() handles correction automatically
        # - Accumulation code must apply: true_start = position - (6 * rotation)
        #   (because it needs absolute baseband position, not chunk-relative)
        
        return {
            "found": (best["position"] >= 0) and (best["correlation"] >= corr_thresh),
            **best
        }
    
    def _extract_packet(self, soft_bits: np.ndarray, sync_info: Dict) -> Optional[np.ndarray]:
        """Extract 258-bit packet at the TRUE packet start.
        
        CRITICAL: Apply jigsaw correction for shifted sync patterns!
        The position from _find_sync() is where the SHIFTED pattern was detected.
        We must correct backwards to find the TRUE packet start.
        
        GUARD: This is the ONLY place rotation correction should happen for extraction.
        """
        detected_pos = sync_info["position"]
        rotation = sync_info.get("sync_rotation", 0)  # Actual rotation amount
        polarity = sync_info["polarity"]
        
        # GUARD ASSERTION: Ensure we're not receiving pre-corrected positions
        # If someone already did "true_start = pos - 6*rotation", this will fail
        # because detected_pos would be too small
        assert rotation >= 0 and rotation <= 29, f"Invalid rotation: {rotation}"
        
        # Jigsaw correction: rotation tells us how much earlier the true start is
        correction = 6 * rotation
        true_start = detected_pos - correction
        
        samples = np.asarray(soft_bits).flatten()
        
        if true_start < 0 or true_start + 258 > len(samples):
            return None
        
        # Extract from corrected position and apply polarity
        pkt = np.asarray(samples[true_start:true_start+258], dtype=float) * polarity
        
        return pkt
    
    def _extract_packet_jigsaw(self, soft_bits: np.ndarray, start_pos: int, 
                               shift: int, polarity: int = 1) -> Optional[np.ndarray]:
        """Extract 258-bit packet with jigsaw assembly for shifted sync patterns.
        
        Per PSK2K spec page 18: "the ping may start somewhere in a packet and end 
        in the next packet. This situation is detected with the shifted synchronization 
        patterns, and the packet bits then are read partly from the first packet and 
        the rest from the subsequent packet."
        
        Args:
            soft_bits: Demodulated soft bit stream
            start_pos: Start position in bit stream
            shift: Sync pattern shift (0, 14, or 29)
            polarity: +1 or -1 for phase ambiguity
            
        Returns:
            258 soft bits, possibly assembled from two packet boundaries
        """
        samples = np.asarray(soft_bits).flatten()
        
        if shift == 0:
            # No shift - extract normally
            if start_pos + 258 > len(samples):
                return None
            return samples[start_pos:start_pos + 258] * polarity
        
        else:
            # Shifted pattern: assemble from tail of one packet + head of next
            # The shift tells us how many bits into the pattern we are
            bits_from_first = 258 - shift
            bits_from_second = shift
            
            if start_pos + 258 > len(samples):
                return None
                
            # Take last (258-shift) bits from current position
            # and first (shift) bits from (start_pos + 258-shift) 
            packet = np.zeros(258, dtype=np.float32)
            packet[:bits_from_first] = samples[start_pos:start_pos + bits_from_first]
            
            next_start = start_pos + bits_from_first
            if next_start + bits_from_second <= len(samples):
                packet[bits_from_first:] = samples[next_start:next_start + bits_from_second]
            else:
                return None  # Can't complete jigsaw
            
            return packet * polarity
    
    
    def _try_decode_format1(self, packet_soft: np.ndarray) -> Dict:
        """Try to decode as Format 1 using soft bits"""
        packet_hard = (packet_soft > 0).astype(int)
        
        # Use SOFT deinterleaving to preserve soft information
        sync_soft, addr_soft, poly1_soft, poly2_soft = deinterleave_format1_soft(packet_soft)
        
        # Also get hard bits for address checking
        sync_hard, addr_hard, poly1_hard, poly2_hard = deinterleave_format1(packet_hard)
        
        # Check address using SOFT correlation (per PSK2K spec page 18)
        # "the 49 soft address bits of this packet are correlated with both,
        # the general address and (if there is a ToCall) the actual private address"
        
        # Convert general address from {0,1} to {-1,+1} for correlation
        general_addr_signed = (self.general_address * 2 - 1).astype(np.float32)
        
        # Compute normalized correlation
        addr_norm = np.linalg.norm(addr_soft)
        if addr_norm > 0:
            soft_correlation = np.dot(addr_soft, general_addr_signed) / addr_norm / np.sqrt(len(addr_soft))
        else:
            soft_correlation = 0.0
        
        # Threshold for general address detection
        # At SNR > -3dB we should get correlation > 0.15
        # At SNR > 0dB we should get correlation > 0.25
        # At SNR > +3dB we should get correlation > 0.40
        is_general = soft_correlation >= 0.15
        
        # If it's clearly not a general address, check if it's private to us
        if not is_general:
            # Check if it's for us (if we have my_callsign)
            if self.my_callsign:
                my_addr = self.source_decoder.encode_callsign(self.my_callsign)[:49]
                private_match = np.sum(addr_hard == my_addr)
                is_for_me = private_match >= 45
                if not is_for_me:
                    return {'success': False, 'error': 'Private address not for me'}
                # It IS for us, so continue to decode it
            else:
                # We don't have my_callsign, so we can't verify private addresses
                # Reject it and let Format 2 try
                return {'success': False, 'error': 'Private address detected, need my_callsign'}
        
        # Combine SOFT polynomial bits for Viterbi
        coded_soft = np.zeros(166)
        coded_soft[0::2] = poly1_soft
        coded_soft[1::2] = poly2_soft
        
        # Normalize soft bits to reasonable range for Viterbi
        # Don't clip - use actual soft decision values
        # Viterbi works better with proper likelihood information
        max_abs = np.max(np.abs(coded_soft))
        if max_abs > 0:
            coded_soft = coded_soft / max_abs * 3.0  # Scale to ±3 range
        
        # Decode using soft bits
        info_bits, _ = self.viterbi.decode_format1(coded_soft)
        
        # Parse message
        if is_general:
            # Format 1 general: callsign (54) + type (2) + parity (15) = 71 bits
            callsign = self.source_decoder.decode_callsign(info_bits[:54])
            
            # Validate callsign first
            if not callsign or len(callsign) < 3:
                return {'success': False, 'error': f'Invalid callsign: {callsign}'}
            
            valid_chars = set('/ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if not all(c in valid_chars for c in callsign):
                return {'success': False, 'error': f'Callsign has invalid characters: {callsign}'}
            
            # PARITY VERIFICATION for Format 1 general CQ
            # Source: callsign (54) + type (2) = 56 bits
            # Parity: 15 bits generated with r=32749
            type_bits = info_bits[54:56]
            expected_parity = self.source_decoder.generate_parity(info_bits[:56], 32749, 15)
            received_parity = info_bits[56:71]
            
            # Check if parity matches
            if not np.array_equal(expected_parity, received_parity):
                return {'success': False, 'error': f'Parity check failed for general CQ (callsign: {callsign})'}
            
            return {
                'success': True,
                'format': 'format1',
                'address_type': 'general',
                'from_call': callsign,
                'message_type': 'CQ',
                'text': f"CQ de {callsign}"
            }
        else:
            report_bits = info_bits[:3]
            report_map = {'000': '', '001': '26', '010': '27', '011': '37'}
            report_str = ''.join(map(str, report_bits))
            report = report_map.get(report_str, '')
            
            callsign = self.source_decoder.decode_callsign(info_bits[3:57])
            
            # Validate callsign - reject if it contains invalid characters or is empty
            if not callsign or len(callsign) < 3:
                return {'success': False, 'error': f'Invalid callsign: {callsign}'}
            
            # Check if callsign contains only valid characters
            valid_chars = set('/ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if not all(c in valid_chars for c in callsign):
                return {'success': False, 'error': f'Callsign has invalid characters: {callsign}'}
            
            # PARITY VERIFICATION (per spec section 6.3)
            # For private Format 1: info_bits (57) + their_call_bits (54) → 14 parity bits, r=16381
            to_call = self.source_decoder.decode_private_address(addr_hard)
            
            # MUST be able to decode TO callsign for private messages
            if not to_call or to_call == "ERROR":
                return {'success': False, 'error': 'Cannot decode private address'}
            
            # ALWAYS verify parity for private messages
            their_call_bits = self.source_decoder.encode_callsign(to_call)
            parity_source = np.concatenate([info_bits[:57], their_call_bits])
            expected_parity = self.source_decoder.generate_parity(parity_source, 16381, 14)
            received_parity = info_bits[57:71]
            
            # Check if parity matches
            if not np.array_equal(expected_parity, received_parity):
                return {'success': False, 'error': 'Parity check failed for private Format 1'}
            
            # Build text
            if report:
                text = f"{to_call} de {callsign} {report}"
            else:
                text = f"{to_call} de {callsign}"
            
            return {
                'success': True,
                'format': 'format1',
                'address_type': 'private',
                'from_call': callsign,
                'to_call': to_call,
                'report': report,
                'message_type': 'CALL',
                'text': text
            }
    
    def _try_decode_format2(self, packet_soft: np.ndarray) -> Dict:
        """Try to decode as Format 2 using soft bits"""
        packet_hard = (packet_soft > 0).astype(int)
        
        # Use SOFT deinterleaving to preserve soft information
        sync_soft, addr_soft, poly_dict_soft = deinterleave_format2_soft(packet_soft)
        
        # Also get hard bits for address checking
        sync_hard, addr_hard, poly_dict_hard = deinterleave_format2(packet_hard)
        
        # Check address (use first 49 bits of callsign encoding)
        # Use fuzzy matching like Format 1 (>=45 bits match)
        if self.my_callsign:
            my_addr = self.source_decoder.encode_callsign(self.my_callsign)[:49]
            addr_match = np.sum(addr_hard == my_addr)
            is_for_me = addr_match >= 45
            if not is_for_me:
                return {'success': False, 'error': f'Not addressed to me (match={addr_match}/49)'}
        
        # Combine SOFT polynomials for Viterbi (preserving soft information!)
        coded_soft = np.zeros(162)
        poly_names = ['Pa', 'Pb', 'Pc', 'Pd', 'Pe', 'Pf', 'Pg', 'Ph', 'Pi']
        for i in range(18):
            for j, poly_name in enumerate(poly_names):
                coded_soft[i*9 + j] = poly_dict_soft[poly_name][i]
        
        # CRITICAL: Scale soft bits for Viterbi decoder
        # Use sign-preserving scaling - clip magnitude but keep sign
        coded_soft = np.sign(coded_soft) * np.minimum(np.abs(coded_soft), 1.5)
        
        # Decode using soft bits
        info_bits, _ = self.viterbi.decode_format2(coded_soft)
        
        # Parse message code (first 3 bits)
        message_code = int(''.join(map(str, info_bits[:3])), 2)
        
        # All 3-bit message codes (0b000-0b111) are valid
        
        # PARITY VERIFICATION (per spec sections 6.4 and 7.4.2)
        # Format 2: message (3 bits) + parity (15 bits) = 18 bits
        # Parity generated from: message_bits + their_callsign + my_callsign
        # Using r=32749 (15 bits)
        
        if not self.my_callsign or not self.partner_callsign:
            # Can't verify parity without both callsigns - reject
            return {'success': False, 'error': 'Format 2 requires both callsigns for parity check'}
        
        # Generate expected parity (order per spec 7.4.2: message + their_call + my_call)
        # NOTE: "their" and "my" are from the TRANSMITTER's perspective!
        # So if we're receiving from partner_callsign, we need:
        #   message + my_callsign (their "their") + partner_callsign (their "my")
        my_call_bits = self.source_decoder.encode_callsign(self.my_callsign)
        their_call_bits = self.source_decoder.encode_callsign(self.partner_callsign)
        # The TRANSMITTER encodes parity as: [message, their_call, my_call]
        # From receiver's perspective: their_call = us (my_callsign), my_call = them (partner_callsign)
        # So we need: [message, my_call_bits, their_call_bits]
        parity_source = np.concatenate([info_bits[:3], my_call_bits, their_call_bits])
        expected_parity = self.source_decoder.generate_parity(parity_source, 32749, 15)
        received_parity = info_bits[3:18]
        
        # Check if parity matches
        if not np.array_equal(expected_parity, received_parity):
            return {'success': False, 'error': 'Format 2 parity check failed'}
        
        message_map = {0b000: 'R26', 0b001: 'R27', 0b010: 'R28', 0b011: 'R29', 0b100: 'R36', 0b101: 'R37', 0b110: 'RR', 0b111: '73'}
        message_text = message_map.get(message_code, 'unknown')
        
        return {
            'success': True,
            'format': 'format2',
            'address_type': 'private',
            'from_call': self.partner_callsign,
            'to_call': self.my_callsign,
            'message_code': message_code,
            'message_type': message_text,
            'text': message_text
        }


# Keep PSK2kReceiver as alias for backward compatibility
PSK2kReceiver = MSK2KReceiver


# =============================================================================
# PART 7: HIGH-LEVEL TRANSMITTER
# From: generate_qso_signals.py (123 lines)
# =============================================================================

class MSK2KTransmitter:
    """High-level MSK2K signal generator"""
    
    def __init__(self, sample_rate: int = 48000):
        """Initialize transmitter"""
        self.sample_rate = sample_rate
        self.source_encoder = PSK2kSourceEncoder()
        self.conv_encoder = PSK2kConvolutionalEncoder()
        self.modulator = MSK2KModulator(sample_rate=sample_rate)
        
        # Sync pattern
        self.sync_pattern = np.array([0,1,0,0,1,0,1,0,0,1,1,1,0,1,1,1,1,1,0,0,
                                      0,1,0,1,1,1,0,0,0,0,0,1,0,0,0,1,1,0,1,0,1,1,0])
        
        # General address
        pattern_str = "1101010000001111100110011011011011011011011011011"
        self.general_address = np.array([int(b) for b in pattern_str], dtype=int)
    
    def generate_cq(self, my_call: str) -> np.ndarray:
        """Generate CQ message"""
        call_bits = self.source_encoder.encode_callsign(my_call)
        type_bits = np.array([0, 1], dtype=int)
        source = np.concatenate([call_bits, type_bits])
        
        parity = self.source_encoder.generate_parity(source, 32749, 15)
        info_bits = np.concatenate([call_bits, type_bits, parity])
        
        coded = self.conv_encoder.encode_format1(info_bits)
        packet = interleave_format1(self.sync_pattern, self.general_address, 
                                    coded[0::2], coded[1::2])
        
        return self._modulate_packet(packet)
    
    def generate_cold_call(self, my_call: str, their_call: str) -> np.ndarray:
        """Generate cold call without report (before receiving from them)
        
        Uses report code '000' which decodes to empty string - proper for initial calls
        when you haven't yet received anything from the other station.
        """
        call_bits = self.source_encoder.encode_callsign(my_call)
        # '000' = no report (cold call before hearing them)
        report_bits = np.array([0, 0, 0], dtype=int)
        
        info_bits = np.concatenate([report_bits, call_bits])
        their_call_bits = self.source_encoder.encode_callsign(their_call)
        
        parity_source = np.concatenate([info_bits, their_call_bits])
        parity = self.source_encoder.generate_parity(parity_source, 16381, 14)
        
        info_with_parity = np.concatenate([info_bits, parity])
        coded = self.conv_encoder.encode_format1(info_with_parity)
        
        # Use proper private address (first 49 bits of generate_private_address)
        private_addr = self.source_encoder.generate_private_address(their_call)[:49]
        packet = interleave_format1(self.sync_pattern, private_addr, 
                                    coded[0::2], coded[1::2])
        
        return self._modulate_packet(packet)
    
    def generate_call_with_report(self, my_call: str, their_call: str, report: str = '26') -> np.ndarray:
        """Generate call with report"""
        call_bits = self.source_encoder.encode_callsign(my_call)
        report_map = {'26': '001', '27': '010', '37': '011'}
        report_bits = np.array([int(b) for b in report_map.get(report, '001')], dtype=int)
        
        info_bits = np.concatenate([report_bits, call_bits])
        their_call_bits = self.source_encoder.encode_callsign(their_call)
        
        parity_source = np.concatenate([info_bits, their_call_bits])
        parity = self.source_encoder.generate_parity(parity_source, 16381, 14)
        
        info_with_parity = np.concatenate([info_bits, parity])
        coded = self.conv_encoder.encode_format1(info_with_parity)
        
        # Use proper private address (first 49 bits of generate_private_address)
        private_addr = self.source_encoder.generate_private_address(their_call)[:49]
        packet = interleave_format1(self.sync_pattern, private_addr, 
                                    coded[0::2], coded[1::2])
        
        return self._modulate_packet(packet)
    
    def generate_format2_message(self, my_call: str, their_call: str, message_type: str) -> np.ndarray:
        """Generate Format 2 message (R-report, RR, RRR, or 73)"""
        # 3-bit message codes for Format 2
        # Reports 26-29 and 36-37 indicate signal quality/workability
        # RRR = roger (alternative acknowledgment)
        # 73 = end of QSO
        message_map = {
            'R26': 0b000,  # Report: 26 (barely decodable)
            'R27': 0b001,  # Report: 27 (weak but solid)
            'R28': 0b010,  # Report: 28 (good signal)
            'R29': 0b011,  # Report: 29 (strong signal)
            'R36': 0b100,  # Report: 36 (very strong/consistent)
            'R37': 0b101,  # Report: 37 (excellent)
            'RR':  0b110,  # Roger Roger (acknowledgment)
            'RRR': 0b110,  # Roger (alias for RR)
            '73':  0b111   # Best regards (end of QSO)
        }
        message_code = message_map.get(message_type, 0b000)
        message_bits = np.array([int(b) for b in format(message_code, '03b')], dtype=int)
        
        my_call_bits = self.source_encoder.encode_callsign(my_call)
        their_call_bits = self.source_encoder.encode_callsign(their_call)
        
        parity_source = np.concatenate([message_bits, their_call_bits, my_call_bits])
        parity = self.source_encoder.generate_parity(parity_source, 32749, 15)
        
        info_bits = np.concatenate([message_bits, parity])
        coded = self.conv_encoder.encode_format2(info_bits)
        
        poly_dict = {
            'Pa': coded[0::9], 'Pb': coded[1::9], 'Pc': coded[2::9],
            'Pd': coded[3::9], 'Pe': coded[4::9], 'Pf': coded[5::9],
            'Pg': coded[6::9], 'Ph': coded[7::9], 'Pi': coded[8::9]
        }
        
        # Use first 49 bits of callsign as private address (matching working code)
        private_addr = self.source_encoder.encode_callsign(their_call)[:49]
        packet = interleave_format2(self.sync_pattern, private_addr, poly_dict)
        
        return self._modulate_packet(packet)
    
    # Alias for backward compatibility with engine_server.py
    def generate_call(self, my_call: str, their_call: str) -> np.ndarray:
        """Generate call (alias for generate_call_with_report)"""
        return self.generate_call_with_report(my_call, their_call, '26')
    
    def _modulate_packet(self, packet: np.ndarray) -> np.ndarray:
        """Modulate packet to MSK audio"""
        return self.modulator.generate_packet_audio(packet)


# Keep PSK2kTransmitter as alias for backward compatibility
PSK2kTransmitter = MSK2KTransmitter


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_test_qso(my_call: str = 'GW4WND', their_call: str = 'DJ5HG',
                     sample_rate: int = 48000, output_dir: str = '.') -> List[str]:
    """
    Generate complete test QSO sequence with MSK2K
    
    Returns list of generated filenames
    """
    from pathlib import Path
    
    tx = MSK2KTransmitter(sample_rate=sample_rate)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    files = []
    
    # 1. CQ
    audio = tx.generate_cq(my_call)
    filename = output_path / '01_cq.wav'
    wavfile.write(str(filename), sample_rate, audio.astype(np.float32))
    files.append(str(filename))
    
    # 2. Call with report
    audio = tx.generate_call_with_report(their_call, my_call, '26')
    filename = output_path / '02_call_26.wav'
    wavfile.write(str(filename), sample_rate, audio.astype(np.float32))
    files.append(str(filename))
    
    # 3. R-report
    audio = tx.generate_format2_message(my_call, their_call, 'R26')
    filename = output_path / '03_r26.wav'
    wavfile.write(str(filename), sample_rate, audio.astype(np.float32))
    files.append(str(filename))
    
    # 4. RR
    audio = tx.generate_format2_message(their_call, my_call, 'RR')
    filename = output_path / '04_rr.wav'
    wavfile.write(str(filename), sample_rate, audio.astype(np.float32))
    files.append(str(filename))
    
    # 5. 73
    audio = tx.generate_format2_message(my_call, their_call, '73')
    filename = output_path / '05_73.wav'
    wavfile.write(str(filename), sample_rate, audio.astype(np.float32))
    files.append(str(filename))
    
    return files


# =============================================================================
# PART 8: SOFT DEINTERLEAVING FUNCTIONS
# =============================================================================

def deinterleave_format1_soft(packet_soft: np.ndarray):
    """
    SOFT Deinterleave Format 1 packet - preserves soft values for Viterbi!
    
    Args:
        packet_soft: 258 soft values (float, typically in range -1 to +1)
        
    Returns:
        sync_soft, addr_soft, poly1_soft, poly2_soft (all float arrays)
    """
    if len(packet_soft) != 258:
        raise ValueError(f"Need 258-element packet, got {len(packet_soft)}")
    
    sync_soft = np.zeros(43, dtype=float)
    addr_soft = np.zeros(49, dtype=float)
    poly1_soft = np.zeros(83, dtype=float)
    poly2_soft = np.zeros(83, dtype=float)
    
    for position, (type_code, index) in enumerate(FORMAT1_TABLE):
        idx = index - 1
        if type_code == 'S':
            sync_soft[idx] = packet_soft[position]
        elif type_code == 'A':
            addr_soft[idx] = packet_soft[position]
        elif type_code == '1':
            poly1_soft[idx] = packet_soft[position]
        elif type_code == '2':
            poly2_soft[idx] = packet_soft[position]
    
    return sync_soft, addr_soft, poly1_soft, poly2_soft


def deinterleave_format2_soft(packet_soft: np.ndarray):
    """
    SOFT Deinterleave Format 2 packet - preserves soft values!
    
    Args:
        packet_soft: 258 soft values
        
    Returns:
        sync_soft, addr_soft, poly_dict_soft
    """
    if len(packet_soft) != 258:
        raise ValueError(f"Need 258-element packet, got {len(packet_soft)}")
    
    sync_soft = np.zeros(43, dtype=float)
    addr_soft = np.zeros(49, dtype=float)
    poly_names = ['Pa', 'Pb', 'Pc', 'Pd', 'Pe', 'Pf', 'Pg', 'Ph', 'Pi']
    poly_dict = {name: np.zeros(18, dtype=float) for name in poly_names}
    
    for position, (type_code, index) in enumerate(FORMAT2_TABLE):
        if type_code == '_':
            continue
        
        idx = index - 1
        type_upper = type_code.upper()
        
        if type_upper == 'S':
            sync_soft[idx] = packet_soft[position]
        elif type_upper == 'A':
            addr_soft[idx] = packet_soft[position]
        else:
            for poly_name in poly_names:
                if type_upper == poly_name.upper():
                    poly_dict[poly_name][idx] = packet_soft[position]
                    break
    
    return sync_soft, addr_soft, poly_dict


# =============================================================================
# PART 9: SOFT ACCUMULATOR FOR WEAK SIGNALS  
# =============================================================================

class PSK2kAccumulator:
    """
    Multi-ping accumulator for weak signal meteor scatter.
    
    This is the key to PSK2k's sensitivity advantage over MSK144/FSK144.
    By accumulating soft bits from multiple weak pings, we can achieve
    3-6 dB better sensitivity than single-ping decode.
    
    Usage:
        acc = PSK2kAccumulator(sample_rate=8000)
        
        # Accumulate soft packets from multiple pings
        for ping_soft in detected_pings:
            acc.accumulate_soft_packet(ping_soft, weight=correlation**2)
        
        # Get combined result
        combined = acc.get_accumulated_packet()
        
        # Decode
        result = acc.decode_accumulated(my_callsign='G2NXX')
    """
    
    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate
        self.receiver = PSK2kReceiver(sample_rate=sample_rate)
        self.viterbi = PSK2kViterbiDecoder()
        self.source_decoder = PSK2kSourceEncoder()
        
        # Accumulation state
        self.accumulated_soft = None
        self.accumulated_weight = None
        self.num_pings = 0
        
        # Known patterns
        self.sync_pattern = np.array([
            0,1,0,0,1,0,1,0,0,1,1,1,0,1,1,1,1,1,0,0,
            0,1,0,1,1,1,0,0,0,0,0,1,0,0,0,1,1,0,1,0,1,1,0
        ])
        general_pattern = "1101010000001111100110011011011011011011011011011"
        self.general_address = np.array([int(b) for b in general_pattern], dtype=int)
    
    def reset(self):
        """Reset accumulator state"""
        self.accumulated_soft = np.zeros(258, dtype=float)
        self.accumulated_weight = np.zeros(258, dtype=float)  # Per-bit weights
        self.num_pings = 0
    
    def accumulate_soft_packet(self, packet_soft: np.ndarray, weight: float = 1.0, 
                              valid_mask: np.ndarray = None, conf_override: np.ndarray = None):
        """
        Accumulate a soft packet with per-bit confidence weighting and valid-bit masking.
        
        CRITICAL FOR MSK: Use magnitude-based confidence, not abs(soft_bits)!
        In MSK, differential phase has large magnitude even in pure noise.
        Only the baseband magnitude indicates signal presence.
        
        Args:
            packet_soft: 258 soft symbols (float values)
            weight: Global quality weight for this packet (e.g., canonical_corr**2)
            valid_mask: Optional 258-element boolean mask of valid bits
            conf_override: Optional 258-element confidence metric (e.g., symbol magnitude)
        """
        if len(packet_soft) != 258:
            raise ValueError(f"Expected 258 soft symbols, got {len(packet_soft)}")
        
        if self.accumulated_soft is None:
            self.reset()
        
        p = np.array(packet_soft, dtype=float)
        
        # Default: if no mask passed, estimate from packet itself
        if valid_mask is None:
            valid = self._estimate_valid_mask(p).astype(float)
        else:
            valid = np.asarray(valid_mask, dtype=bool).astype(float)
            if valid.shape[0] != 258:
                raise ValueError(f"Expected valid_mask length 258, got {valid.shape[0]}")
        
        # Per-bit confidence
        # For MSK: use magnitude override (signal presence), not abs(soft_bits)!
        if conf_override is not None:
            conf = np.asarray(conf_override, dtype=float)
            if conf.shape[0] != 258:
                raise ValueError(f"Expected conf_override length 258, got {conf.shape[0]}")
        else:
            conf = np.abs(p)  # Fallback for PSK
        
        # Combined per-bit weight = global_weight * local_confidence * valid_mask
        # Only valid bits contribute!
        w = float(weight) * conf * valid
        
        self.accumulated_soft += w * p
        self.accumulated_weight += w
        self.num_pings += 1
    
    def _estimate_valid_mask(self, packet: np.ndarray) -> np.ndarray:
        """
        Heuristic gating: marks bits valid if |soft| is meaningfully above noise floor.
        
        Strategy: Use a FIXED threshold relative to packet median, not Q25.
        In clean signals, Q25 is too high. In noisy signals with some valid bits,
        median separates signal from noise better.
        """
        a = np.abs(packet)
        
        # Use median as reference (more robust than Q25 for mixed signal/noise)
        median = float(np.median(a))
        
        # Threshold: 70% of median
        # Clean signal: median~0.7 → thr=0.49 (accepts most bits)
        # Noisy signal with some valid: median~0.4 → thr=0.28 (accepts strong bits)
        mult = getattr(self, "valid_median_mult", 0.7)
        floor = getattr(self, "valid_abs_floor", 0.05)
        thr = max(floor, median * mult)
        mask = a >= thr
        
        # Optional: require at least some minimum number of valid bits
        # At low SNR, even 40-60 bits (15-23%) can be useful for accumulation
        min_valid = getattr(self, "min_valid_bits", 40)
        if int(np.sum(mask)) < min_valid:
            # If it's *too* weak, return all-false so it contributes nothing
            return np.zeros(258, dtype=bool)
        
        return mask
    
    def get_accumulated_packet(self) -> np.ndarray:
        """
        Get the accumulated soft packet (weighted average per bit).
        
        Returns:
            258 soft symbols representing the accumulated packet
        """
        if self.accumulated_soft is None or self.num_pings == 0:
            return np.zeros(258, dtype=float)
        
        # Elementwise division with protection against zero weights
        return self.accumulated_soft / (self.accumulated_weight + 1e-12)
    
    def decode_accumulated(self, my_callsign: str = None, 
                          partner_callsign: str = None) -> Dict:
        """
        Decode the accumulated soft packet using SOFT deinterleaving.
        
        This is the key improvement - soft values are preserved all the way
        through to the Viterbi decoder, not quantised to hard bits first.
        
        Args:
            my_callsign: Receiver's callsign (for address checking)
            partner_callsign: QSO partner's callsign (for Format 2)
            
        Returns:
            Decode result dictionary
        """
        packet_soft = self.get_accumulated_packet()
        
        # SOFT deinterleave - preserves soft information!
        sync_soft, addr_soft, poly1_soft, poly2_soft = deinterleave_format1_soft(packet_soft)
        
        # Hard decision on address only (for checking)
        addr_hard = (addr_soft > 0).astype(int)
        
        # Check if general (CQ) or private address using SOFT correlation
        # More robust than hard bit matching at low SNR
        general_addr_signed = (self.general_address * 2 - 1).astype(np.float32)
        addr_norm = np.linalg.norm(addr_soft)
        if addr_norm > 0:
            soft_correlation = np.dot(addr_soft, general_addr_signed) / addr_norm / np.sqrt(len(addr_soft))
        else:
            soft_correlation = 0.0
        
        is_general = soft_correlation >= 0.15
        
        if not is_general and my_callsign:
            my_addr = self.source_decoder.encode_callsign(my_callsign)[:49]
            is_for_me = np.array_equal(addr_hard, my_addr)
            if not is_for_me:
                # Try Format 2
                return self._decode_format2_soft(packet_soft, my_callsign, partner_callsign)
        
        # Combine polynomial soft bits for Viterbi (SOFT - this is key!)
        coded_soft = np.zeros(166, dtype=float)
        coded_soft[0::2] = poly1_soft
        coded_soft[1::2] = poly2_soft
        
        # CRITICAL: Scale soft bits for Viterbi decoder
        # Use sign-preserving scaling - clip magnitude but keep sign
        coded_soft = np.sign(coded_soft) * np.minimum(np.abs(coded_soft), 1.5)
        
        # Decode with full soft information
        info_bits, path_metric = self.viterbi.decode_format1(coded_soft)
        
        # Parse message
        result = {
            'success': True,
            'format': 'format1',
            'num_pings_accumulated': self.num_pings,
            'effective_snr_gain_db': 10 * np.log10(max(1, self.num_pings)),
            'path_metric': path_metric
        }
        
        if is_general:
            callsign = self.source_decoder.decode_callsign(info_bits[:54])
            
            # Validate callsign
            if not callsign or len(callsign) < 3:
                return {'success': False, 'error': f'Invalid callsign: {callsign}'}
            
            valid_chars = set('/ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if not all(c in valid_chars for c in callsign):
                return {'success': False, 'error': f'Callsign has invalid characters: {callsign}'}
            
            # PARITY CHECK for general CQ (per spec 6.3)
            type_bits = info_bits[54:56]
            expected_parity = self.source_decoder.generate_parity(info_bits[:56], 32749, 15)
            received_parity = info_bits[56:71]
            
            if not np.array_equal(expected_parity, received_parity):
                return {'success': False, 'error': 'Accumulator: Parity check failed for general CQ'}
            
            result.update({
                'address_type': 'general',
                'from_call': callsign,
                'message_type': 'CQ',
                'text': f"CQ de {callsign}"
            })
        else:
            report_bits = info_bits[:3]
            report_map = {'001': '26', '010': '27', '011': '37'}
            report_str = ''.join(map(str, report_bits))
            report = report_map.get(report_str, 'unknown')
            
            callsign = self.source_decoder.decode_callsign(info_bits[3:57])
            
            # Validate callsign
            if not callsign or len(callsign) < 3:
                return {'success': False, 'error': f'Invalid callsign: {callsign}'}
            
            valid_chars = set('/ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if not all(c in valid_chars for c in callsign):
                return {'success': False, 'error': f'Callsign has invalid characters: {callsign}'}
            
            # PARITY CHECK for private Format 1 (per spec 6.3)
            # Decode TO address from address bits
            to_call = self.source_decoder.decode_private_address(addr_hard)
            
            if not to_call or to_call == "ERROR":
                return {'success': False, 'error': 'Accumulator: Cannot decode private address'}
            
            # Verify parity
            their_call_bits = self.source_decoder.encode_callsign(to_call)
            parity_source = np.concatenate([info_bits[:57], their_call_bits])
            expected_parity = self.source_decoder.generate_parity(parity_source, 16381, 14)
            received_parity = info_bits[57:71]
            
            if not np.array_equal(expected_parity, received_parity):
                return {'success': False, 'error': 'Accumulator: Parity check failed for private Format 1'}
            
            result.update({
                'address_type': 'private',
                'from_call': callsign,
                'to_call': to_call,
                'report': report,
                'message_type': 'CALL',
                'text': f"{to_call} de {callsign} {report}"
            })
        
        return result
    
    def _decode_format2_soft(self, packet_soft: np.ndarray, 
                            my_callsign: str, partner_callsign: str) -> Dict:
        """Decode Format 2 with soft deinterleaving"""
        sync_soft, addr_soft, poly_dict_soft = deinterleave_format2_soft(packet_soft)
        
        # Check address
        addr_hard = (addr_soft > 0).astype(int)
        if my_callsign:
            my_addr = self.source_decoder.encode_callsign(my_callsign)[:49]
            is_for_me = np.array_equal(addr_hard, my_addr)
            if not is_for_me:
                return {'success': False, 'error': 'Not addressed to me'}
        
        # Combine polynomials (SOFT!)
        coded_soft = np.zeros(162, dtype=float)
        poly_names = ['Pa', 'Pb', 'Pc', 'Pd', 'Pe', 'Pf', 'Pg', 'Ph', 'Pi']
        for i in range(18):
            for j, poly_name in enumerate(poly_names):
                coded_soft[i * 9 + j] = poly_dict_soft[poly_name][i]
        
        # CRITICAL: Scale soft bits for Viterbi decoder
        coded_soft = np.sign(coded_soft) * np.minimum(np.abs(coded_soft), 1.5)
        
        # Decode
        info_bits, path_metric = self.viterbi.decode_format2(coded_soft)
        
        # Parse message code
        message_code = int(''.join(map(str, info_bits[:3])), 2)
        message_map = {0b000: 'R26', 0b001: 'R27', 0b010: 'R28', 0b011: 'R29', 0b100: 'R36', 0b101: 'R37', 0b110: 'RR', 0b111: '73'}
        message_text = message_map.get(message_code, 'unknown')
        
        return {
            'success': True,
            'format': 'format2',
            'address_type': 'private',
            'from_call': partner_callsign if partner_callsign else 'UNKNOWN',
            'to_call': my_callsign,
            'message_code': message_code,
            'message_type': message_text,
            'text': message_text,
            'num_pings_accumulated': self.num_pings,
            'effective_snr_gain_db': 10 * np.log10(max(1, self.num_pings)),
            'path_metric': path_metric
        }
    
    def get_stats(self) -> Dict:
        """Get accumulation statistics"""
        if self.accumulated_weight is None:
            return {'num_pings': 0, 'coverage': 0, 'coverage_percent': 0, 'snr_gain_db': 0}
        
        coverage = int(np.sum(self.accumulated_weight > 0))
        return {
            'num_pings': self.num_pings,
            'coverage': coverage,
            'coverage_percent': 100 * coverage / 258,
            'effective_snr_gain_db': 10 * np.log10(max(1, self.num_pings))
        }


# =============================================================================
# MAIN - Example usage
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("PSK2k Complete Implementation - SOFT ACCUMULATION")
    print("Unified Module with Soft Deinterleaving for Weak Signal Operation")
    print("=" * 70)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        my_call = sys.argv[2] if len(sys.argv) > 2 else 'GW4WND'
        their_call = sys.argv[3] if len(sys.argv) > 3 else 'DJ5HG'
        
        print(f"\nGenerating test QSO: {my_call} <-> {their_call}")
        files = generate_test_qso(my_call, their_call, output_dir='test_qso')
        print(f"\nGenerated {len(files)} files:")
        for f in files:
            print(f"  {f}")
    else:
        print("\nUsage:")
        print("  python psk2k_complete.py generate [CALL1] [CALL2]")
        print("\nOr import as module:")
        print("  from msk2k_complete import MSK2KTransmitter, MSK2KReceiver, PSK2kAccumulator")
        print("\nSoft Accumulation Example:")
        print("  acc = PSK2kAccumulator(sample_rate=8000)")
        print("  for ping_soft in detected_pings:")
        print("      acc.accumulate_soft_packet(ping_soft, weight=correlation**2)")
        print("  result = acc.decode_accumulated(my_callsign='G2NXX')")

