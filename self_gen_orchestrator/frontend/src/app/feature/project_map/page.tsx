"use client";

import React, { useState, useEffect } from "react";
import { FileTree, FileEntry } from "@/components/file-tree/file-tree";

export default function ProjectMapPage() {
  const [projects, setProjects] = useState<string[]>([]);

  useEffect(() => {
    fetch("/projects.json")
      .then((res) => res.json())
      .then(setProjects)
      .catch(console.error);
  }, []);

  const treeData: FileEntry[] = projects.map((name) => ({
    name,
    path: name,
    type: "file",
    content: "",
  }));

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Projects Mind Map</h1>
      <FileTree basePath="" files={treeData} onFileSelect={() => {}} />
    </div>
  );
}