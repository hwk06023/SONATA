#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');
const removeMd = require('remove-markdown');

const DOCS_DIR = path.join(__dirname, '../../');
const OUTPUT_PATH = path.join(__dirname, 'search-data.json');

function buildSearchIndex() {
  const searchData = [];
  const allFiles = getMarkdownFiles(DOCS_DIR);
  
  allFiles.forEach(file => {
    const fileContent = fs.readFileSync(file, 'utf8');
    const { data, content } = matter(fileContent);
    
    if (data.nav_exclude) return;
    
    const urlPath = getUrlPath(file);
    
    searchData.push({
      title: data.title || getDefaultTitle(file),
      content: processContent(content),
      url: urlPath
    });
  });
  
  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(searchData, null, 2));
  console.log(`Search data generated at ${OUTPUT_PATH}`);
}

function getMarkdownFiles(dir) {
  let results = [];
  const items = fs.readdirSync(dir);
  
  items.forEach(item => {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);
    
    if (stat.isDirectory() && !shouldIgnoreDirectory(item)) {
      results = results.concat(getMarkdownFiles(itemPath));
    } else if (isMarkdownFile(item)) {
      results.push(itemPath);
    }
  });
  
  return results;
}

function shouldIgnoreDirectory(dirName) {
  const ignoreDirs = ['.git', 'node_modules', '.jekyll-cache', '_site'];
  return dirName.startsWith('_') || ignoreDirs.includes(dirName);
}

function isMarkdownFile(fileName) {
  return fileName.endsWith('.md') || fileName.endsWith('.markdown');
}

function getUrlPath(filePath) {
  const relativePath = path.relative(DOCS_DIR, filePath);
  const extName = path.extname(relativePath);
  const baseName = path.basename(relativePath, extName);
  const dirName = path.dirname(relativePath);
  
  let urlPath = dirName !== '.' ? `/${dirName}` : '';
  
  // Handle special case for index.md
  if (baseName.toLowerCase() === 'index') {
    return '/SONATA' + urlPath;
  }
  
  return '/SONATA' + urlPath + '/' + baseName;
}

function getDefaultTitle(filePath) {
  const fileName = path.basename(filePath, path.extname(filePath));
  return fileName.split('-').map(word => {
    return word.charAt(0).toUpperCase() + word.slice(1);
  }).join(' ');
}

function processContent(content) {
  // Remove code blocks
  content = content.replace(/```[\s\S]*?```/g, '');
  
  // Remove HTML tags
  content = content.replace(/<[^>]*>/g, '');
  
  // Remove markdown formatting
  content = removeMd(content);
  
  // Remove extra whitespace
  content = content.replace(/\s+/g, ' ').trim();
  
  return content;
}

buildSearchIndex(); 