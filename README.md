# LittleReader

A lightweight hybrid system for parsing and digitizing historical newspaper pages.

## Introduction

Historical newspapers represent an important source of cultural, social, and historical information. The ongoing digitization of newspaper archives has created the opportunity to automatically extract and structure their content through modern computer vision and machine learning techniques. However, historical newspapers remain particularly challenging due to their complex and irregular layouts, degraded scans, and heterogeneous typography.

This repository contains the implementation of *LittleReader*, a lightweight hybrid system for parsing and digitizing historical newspaper pages. The proposed architecture combines a multi-stage pipeline with multimodal approaches in order to detect page layouts, reconstruct reading order, and extract textual content in a structured JSON format. *LittleReader* is composed of three main modules: a layout parser based on RT-DETR, a heuristic-based layout handler designed for occidental newspapers, and an OCR module that integrates classical OCR methods with Vision-Language Models (VLMs).

## Document Layout Detection Module

## Layout Handling Module

## OCR Module
